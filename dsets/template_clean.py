def clean_template_file(input_file, output_file):
    try:

        with open(input_file, 'r', encoding='utf-8') as f:

            lines = [line.strip() for line in f if line.strip()]


        with open(output_file, 'w', encoding='utf-8') as f:

            for line in lines:
                f.write(line + '\n')

        print(f"{output_file}")
        print(f"{len(lines)}")

    except FileNotFoundError:
        print("error")
    except Exception as e:
        print(f"{e}")


clean_template_file('template.txt', 'template_cleaned.txt')