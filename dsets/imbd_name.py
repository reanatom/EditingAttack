import gzip
import time

import os
import requests
from tqdm import tqdm


def download_imdb_file(url, save_path):

    if os.path.exists(save_path):
        return


    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("error")
    else:
        print("success")


def extract_imdb_names(input_gz_path, output_txt_path):

    start_time = time.time()

    extracted_count = 0


    with gzip.open(input_gz_path, 'rt', encoding='utf-8') as f_in, \
            open(output_txt_path, 'w', encoding='utf-8') as f_out:


        header_line = f_in.readline().strip()
        headers = header_line.split('\t')

        if 'primaryName' not in headers:
            raise ValueError("error")

        name_index = headers.index('primaryName')


        for line in f_in:
            columns = line.strip().split('\t')


            if len(columns) > name_index:
                name = columns[name_index]
                if name != r'\N':
                    f_out.write(name + '\n')
                    extracted_count += 1


            if extracted_count % 1000000 == 0:
                print(extracted_count)

    elapsed_time = time.time() - start_time
    print("-" * 30)



if __name__ == "__main__":
    IMDB_URL = "https://datasets.imdbws.com/name.basics.tsv.gz"
    INPUT_FILE = "name.basics.tsv.gz"
    OUTPUT_FILE = "imdb_real_names_pool.txt"

    try:
        download_imdb_file(IMDB_URL, INPUT_FILE)

        extract_imdb_names(INPUT_FILE, OUTPUT_FILE)

    except Exception as e:
        print(f"error: {e}")