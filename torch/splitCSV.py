import os

def split_csv(file_path, n_parts, has_header, output_dir):
    if not isinstance(n_parts, int) or isinstance(n_parts, bool):
        raise Exception("Param 'n_parts' should be a positive integer.")

    if n_parts <= 0:
        raise Exception("Param 'n_parts' should be a positive integer.")

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # First pass: Count total lines
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        total_lines-= 1 if has_header else 0
    
    chunk_size = total_lines // n_parts
    remainder = total_lines % n_parts
    
    # Second pass: Split the file
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline() if has_header else None
        part = 1
        line_count = 0
        
        out_file = open(os.path.join(output_dir, f"{filename}_part_{part}.csv"), 'w', encoding='utf-8')
        if header is not None:
            out_file.write(header)
        
        for line in f:
            if line_count >= chunk_size + (1 if part <= remainder else 0):
                out_file.close()
                part += 1
                line_count = 0
                out_file = open(os.path.join(output_dir, f"{filename}_part_{part}.csv"), 'w', encoding='utf-8')

                if header is not None:
                    out_file.write(header)
            
            out_file.write(line)
            line_count += 1
        out_file.close()
        
    print(f"CSV split into {n_parts} parts successfully in '{output_dir}'")


if __name__ == '__main__':
    split_csv("../raw_data/data_d9_2021_09_02.csv", 10, True, "../raw_data/data-d9-parts")