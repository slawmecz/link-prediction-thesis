from collections import Counter
from pathlib import Path

# Read the file and count types
type_counter = Counter()

with open('entity2type.txt', 'r', encoding='utf-8') as f:
    for line in f:
        # Split the line and skip the first element (entity)
        parts = line.strip().split()
        if len(parts) > 1:
            # Add all types (everything except the first element) to the counter
            types = parts[1:]
            type_counter.update(types)

# Sort types by frequency (most common first)
sorted_types = type_counter.most_common()

# Save the results
output_path = Path('type_frequencies.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    for type_name, count in sorted_types:
        f.write(f'{type_name}\t{count}\n')

print(f'Results have been saved to {output_path}') 