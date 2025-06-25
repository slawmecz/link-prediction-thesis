# Simple script to check entity2type.txt format
with open('entity2type.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 5:  # Show first 5 lines
            print(f"Line {i+1}: {line.strip()}")
        else:
            break 