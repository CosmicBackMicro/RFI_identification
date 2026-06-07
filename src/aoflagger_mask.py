#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib
# 设置非交互式后端，这对多进程和无头环境非常重要，也能稍微提升速度
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import aoflagger as aof
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import argparse
import fitsio
import cv2

def parse_fil_header(filename):
    """
    Parses the header of a Sigproc Filterbank file.
    Returns a dictionary of header keywords and the size of the header in bytes.
    """
    header = {}
    try:
        with open(filename, 'rb') as f:
            # Check for HEADER_START
            # We assume it starts near the beginning.
            # Sigproc headers are variable length.
            
            # Read explicitly in a loop to decode keywords
            def read_string(f):
                len_bytes = f.read(4)
                if len(len_bytes) < 4: return None
                strlen = np.frombuffer(len_bytes, dtype=np.int32)[0]
                if strlen < 0 or strlen > 256: return None # Sanity check
                return f.read(strlen).decode('ascii')

            def read_int(f):
                return np.frombuffer(f.read(4), dtype=np.int32)[0]

            def read_double(f):
                return np.frombuffer(f.read(8), dtype=np.float64)[0]

            # Start parsing
            start = read_string(f)
            if start != "HEADER_START":
                # Maybe retry seeking or handle garbage
                f.seek(0)
                
            while True:
                key = read_string(f)
                if key is None: break
                
                if key == "HEADER_END":
                    header['header_size'] = f.tell()
                    break
                
                # Check known types
                if key in ['source_name', 'rawdatafile']:
                    header[key] = read_string(f)
                elif key in ['machine_id', 'telescope_id', 'data_type', 'nchans', 'nbits', 'nifs', 'scan_number', 'barycentric', 'pulsarcentric']:
                    header[key] = read_int(f)
                elif key in ['tstart', 'tsamp', 'fch1', 'foff', 'refdm', 'az_start', 'za_start', 'src_raj', 'src_dej']:
                    header[key] = read_double(f)
                else:
                    # heuristic fallbacks if unknown key... Sigproc format is tricky without full spec.
                    # Assume double for unknown keys is safest? Or int? 
                    # Usually keys are standard. 
                    # If we hit an unknown key, we might desync. Best effort.
                     header[key] = read_double(f) # Many params are doubles
                     
    except Exception as e:
        print(f"Error parsing filterbank header: {e}")
        return None
        
    return header

def get_file_metadata(filename):
    """
    统一获取文件元数据，支持 .fil 和 .fits
    返回字典: {'type': 'fil'|'fits', 'nchan': int, 'nsamples': int, 'header_size': int, 'n_subints': int}
    """
    ext = os.path.splitext(filename)[1].lower()
    meta = {}
    
    if ext in ['.fits', '.fit']:
        meta['type'] = 'fits'
        with fitsio.FITS(filename, 'r') as f:
            if 'SUBINT' in f:
                hdu = f['SUBINT']
            else:
                hdu = f[1]
            header = hdu.read_header()
            meta['nchan'] = int(header['NCHAN'])
            meta['nbits'] = int(header.get('NBITS', 8)) # Usually packed
            meta['nsblk'] = int(header['NSBLK']) # Samples per subint row
            meta['n_subints'] = hdu.get_nrows()
            meta['tsamp'] = float(header.get('TBIN', header.get('TSAMP', 1.0)))
            # Total samples approx
            meta['total_samples'] = meta['n_subints'] * meta['nsblk']
            
    else:
        # Assume Filterbank
        meta['type'] = 'fil'
        header = parse_fil_header(filename)
        if not header:
            return None
        meta.update(header)
        # Normalize keys
        meta['nchan'] = int(header.get('nchans', header.get('nchan', 0)))
        meta['nsblk'] = 0 # variable
        file_size = os.path.getsize(filename)
        data_size = file_size - meta['header_size']
        bytes_per_sample = (meta['nbits'] // 8) * meta['nchan']
        if bytes_per_sample > 0:
            meta['total_samples'] = data_size // (meta['nbits'] // 8) // meta['nchan']
        else:
             meta['total_samples'] = 0
             
    return meta

def read_psrfits_row(filename, row_idx):
    """读取 PSRFITS 的一行并转换为 (Time, Freq) float32 数组"""
    with fitsio.FITS(filename, 'r') as fits:
        if 'SUBINT' in fits:
            hdu = fits['SUBINT']
        else:
            hdu = fits[1]
        
        # Read header for shape info
        # Optimization: In a long loop, repeated header reads might be slow, 
        # but passing header info around processes is complex. fitsio is fast enough.
        header = hdu.read_header()
        nchan = int(header["NCHAN"])
        nsblk = int(header["NSBLK"])
        
        # Read specific row
        row_data = hdu.read(rows=[row_idx])
        
    record = row_data[0]
    raw_data = np.asarray(record["DATA"])
    
    # Handle dimensions: target (nsblk, nchan) -> (Time, Freq)
    if raw_data.ndim > 1:
        raw_data = raw_data.squeeze()
        
    if raw_data.ndim == 2:
        if raw_data.shape == (nsblk, nchan):
            arr = raw_data.astype(np.float32)
        elif raw_data.shape == (nchan, nsblk):
            arr = raw_data.T.astype(np.float32)
        else:
            arr = raw_data.reshape(nsblk, nchan).astype(np.float32)
    else:
        try:
            arr = raw_data.reshape(nsblk, nchan).astype(np.float32)
        except ValueError:
            arr = raw_data.reshape(nchan, nsblk).T.astype(np.float32)

    # Apply Scale and Offset
    dat_scl = np.asarray(record["DAT_SCL"], dtype=np.float32)
    dat_offs = np.asarray(record["DAT_OFFS"], dtype=np.float32)
    
    # Safe dimension broadcast
    if dat_scl.size >= nchan: dat_scl = dat_scl[:nchan]
    if dat_offs.size >= nchan: dat_offs = dat_offs[:nchan]
        
    arr *= dat_scl[np.newaxis, :]
    arr += dat_offs[np.newaxis, :]
    
    return arr, nchan

def process_chunk(chunk_idx, fil_path, meta, samples_per_subint, strategy_path, output_folder, base_name, no_vis=False):
    """
    处理单个数据块。
    对于 PSRFITS，chunk_idx 对应 Row Index。
    对于 Filterbank，chunk_idx 对应分块索引。
    """
    try:
        if meta['type'] == 'fits':
            # PSRFITS Reader
            data_2d, nchan = read_psrfits_row(fil_path, chunk_idx)
            actual_samples = data_2d.shape[0] # Time samples in this row
        else:
            # Filterbank Reader
            header_size = meta['header_size']
            nchan = meta['nchan']
            nbits = meta['nbits']
            
            with open(fil_path, 'rb') as f:
                offset = header_size + chunk_idx * samples_per_subint * nchan * (nbits // 8)
                f.seek(offset)
                
                read_type = np.float32 
                raw_data = np.fromfile(f, dtype=read_type, count=samples_per_subint * nchan)
            
            actual_samples = raw_data.size // nchan
            if actual_samples == 0: return None
            
            data_2d = raw_data.reshape(actual_samples, nchan)

        # Common AOFlagger Logic
        flagger = aof.AOFlagger()
        try:
            strategy = flagger.load_strategy_file(strategy_path)
        except Exception:
            return f"Error loading strategy: {strategy_path}"
        
        # Prepare Buffer for AOFlagger (Freq, Time)
        width, height = actual_samples, nchan
        data_buffer = np.ascontiguousarray(data_2d.T) # Transpose (Time, Freq) -> (Freq, Time)

        image_set = flagger.make_image_set(width, height, 1)
        image_set.set_image_buffer(0, data_buffer)
        
        mask_set = strategy.run(image_set)
        mask_buffer = mask_set.get_buffer()
        
        # Restore Mask to match Input (Time, Freq) for plotting
        if mask_buffer.ndim == 1:
            mask_2d = mask_buffer.reshape(height, width).T
        else:
            mask_2d = mask_buffer.T
            
        flag_percent = np.sum(mask_2d) / mask_2d.size * 100
        
        # Orient data for visualization: (Freq, Time) with Low Freq at bottom
        # Matches visualize_fits.py and cv2.imwrite layout (Top-Down with flip)
        plot_data = np.flipud(data_2d.T)
        plot_mask = np.flipud(mask_2d.astype(float).T)
        
        # Save 8-bit Mask for AI comparison (0=Clean, 255=RFI)
        # Unified naming convention: {basename}_block{idx}.png
        mask_filename = os.path.join(output_folder, f"{base_name}_block{chunk_idx:04d}.png")
        mask_uint8 = (plot_mask.astype(np.uint8) * 255)
        cv2.imwrite(mask_filename, mask_uint8)
        
        if not no_vis:
            output_png = os.path.join(output_folder, f"{base_name}_block{chunk_idx:04d}_vis.png")
            
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 1, 1)
            d_min, d_max = np.percentile(plot_data, [1, 99])
            plt.imshow(plot_data, aspect='auto', cmap='viridis', vmin=d_min, vmax=d_max)
            plt.title(f"Block/Row {chunk_idx} ({actual_samples} samples) {meta['type'].upper()}")
            
            plt.subplot(2, 1, 2)
            plt.imshow(plot_mask, aspect='auto', cmap='Reds', interpolation='nearest')
            plt.title(f"Mask ({flag_percent:.2f}% flagged)")
            
            plt.tight_layout()
            plt.savefig(output_png)
            plt.close() 
        
        return chunk_idx

    except Exception as e:
        return f"Error in chunk {chunk_idx}: {str(e)}"

def generate_aof_mask(input_path, strategy_path=None, output_folder="masks", no_vis=False):
    print(f"--- Processing: {input_path} ---")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    meta = get_file_metadata(input_path)
    if not meta:
        print("Error: Could not parse file metadata.")
        return

    print(f"File Type: {meta['type']}")
    print(f"Params: nchan={meta['nchan']}, nbits={meta.get('nbits','?')}, tsamp={meta.get('tsamp', 0):.6f}")

    # Determine loop range
    samples_per_subint = 1024 # Fixed default
    if meta['type'] == 'fits':
        num_chunks = meta['n_subints']
        print(f"Processing {num_chunks} rows from PSRFITS table.")
        # Override samples_per_subint for info, though it's fixed by file row size
        samples_per_subint = meta['nsblk'] 
    else:
        # Filterbank logic
        total_samples = meta['total_samples']
        num_chunks = int(np.ceil(total_samples / samples_per_subint))
        print(f"Processing {num_chunks} blocks (Total samples: {total_samples})")

    # Strategy Loading
    if strategy_path is None:
        potential_strategies = [
            'parkes-default.lua',  # Check current directory first
            os.path.join(os.getcwd(), 'parkes-default.lua'),
            '/usr/local/share/aoflagger/strategies/parkes-default.lua',
            '/usr/share/aoflagger/strategies/parkes-default.lua'
        ]
        for s in potential_strategies:
            if os.path.exists(s): strategy_path = s; break
    
    if not strategy_path:
        print("Error: No strategy file found (looked for parkes-default.lua).")
        return
    print(f"Strategy: {strategy_path}")
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Parallel Processing
    # For FITS, parallelism is file-safe as each process opens file efficiently
    max_workers = max(1, (os.cpu_count() or 1) - 2)
    print(f"Starting with {max_workers} processes...")
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(num_chunks):
            futures.append(executor.submit(
                process_chunk,
                i, input_path, meta, samples_per_subint, 
                strategy_path, output_folder, base_name, no_vis
            ))
            
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if isinstance(res, str) and "Error" in res:
                print(f"[FAIL] {res}")
            
            if completed % 10 == 0 or completed == num_chunks:
                elapsed = time.time() - start_time
                speed = completed / elapsed
                print(f"Progress: {completed}/{num_chunks} ({speed:.2f} chunks/s)")

    print(f"Done in {time.time()-start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AOFlagger on data (Filterbank .fil or PSRFITS .fits).")
    parser.add_argument("input_fil", type=str, help="Path to input data file (.fil or .fits).")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory for masks (default: ./results/aoflagger_mask).")
    parser.add_argument("-S", "--strategy", type=str, default=None, help="Path to custom AOFlagger strategy file (.lua). Default: parkes-default.lua")
    parser.add_argument("--no-vis", action="store_true", help="Disable generating visualization plots.")

    args = parser.parse_args()

    input_fil = args.input_fil
    
    # Generate default output path
    if args.output is None:
        default_output = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/aoflagger_mask")
        out_folder = default_output
    else:
        out_folder = args.output
    
    generate_aof_mask(input_fil, strategy_path=args.strategy, output_folder=out_folder, no_vis=args.no_vis)
