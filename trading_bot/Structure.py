import os

def make_corresponding_dir(target_root_dir, current_dir):
    target_dir = os.path.join(target_root_dir, current_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

def copy_empty_dir_structure_from_file(target_root_dir, filenames_file):
    with open(filenames_file, 'r') as file:
        for line in file:
            relative_path = line.strip()
            if relative_path:
                make_corresponding_dir(target_root_dir, relative_path)

def save_dir_structure_to_file(source_root_dir, output_file):
    with open(output_file, 'w') as file:
        for current_dir, dirs, files in os.walk(source_root_dir):
            relative_path = os.path.relpath(current_dir, source_root_dir)
            file.write(relative_path + '\n')
            for dir_name in dirs:
                file.write(os.path.join(relative_path, dir_name) + '\n')

# Démo
source_root_dir = r"c:\Users\timot\OneDrive\Bureau\BOT TRADING BIG 2025\trading_bot"
output_file = r"c:\Users\timot\OneDrive\Bureau\BOT TRADING BIG 2025\trading_bot\filenames.txt"

save_dir_structure_to_file(source_root_dir, output_file)