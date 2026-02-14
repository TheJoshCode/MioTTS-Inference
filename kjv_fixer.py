input_file = "kjv.txt"
output_file = "kjv_clean.txt"

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    for line in infile:
        # Skip header lines
        if "\t" not in line:
            continue
        
        # Split on first tab
        _, verse_text = line.split("\t", 1)
        outfile.write(verse_text.strip() + "\n")

print("Done. Clean text saved to kjv_clean.txt")
