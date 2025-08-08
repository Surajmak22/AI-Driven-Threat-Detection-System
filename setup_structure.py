import os

def create_structure():
    # Base directories
    base_dirs = [
        'src/core',
        'src/models',
        'src/utils',
        'src/features',
        'src/detection',
        'src/visualization',
        'src/config',
        'tests',
        'data/raw',
        'data/processed',
        'docs'
    ]
    
    # Create each directory
    for directory in base_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files for Python packages
    for root, dirs, _ in os.walk('src'):
        for d in dirs:
            init_file = os.path.join(root, d, '__init__.py')
            open(init_file, 'a').close()
            print(f"Created: {init_file}")
    
    print("\nProject structure created successfully!")

if __name__ == "__main__":
    create_structure()
