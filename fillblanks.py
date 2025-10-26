import csv

input_file = 'datasets/stock prices.csv'
output_file = 'datasets/stock prices modified.csv'

with open(input_file, 'r', newline='') as infile, \
     open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        new_row = []
        for cell in row:
            if cell == '':  # Check for empty string
                new_row.append('0')
            else:
                new_row.append(cell)
        writer.writerow(new_row)

print(f"Empty cells in '{input_file}' replaced with '0' and saved to '{output_file}'.")