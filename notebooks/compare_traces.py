from dataclasses import dataclass
from typing import List
import argparse

@dataclass
class Component:
    name: str
    type: str
    n_visits: int
    input_str: str
    output_str: str

def remove_reset_color(text):
    return text.replace("\033[0m", "")

def parse_log_file(file_path: str) -> List[Component]:
    components = []
    current_component = {}
    
    with open(file_path, 'r') as f:
        
        for line in f:
            line = remove_reset_color(line)

            if 'Operation: haystack.component.run' in line:
                if current_component:
                    components.append(Component(**current_component))
                    current_component = {}
                continue
            
            # Parse component properties
            if 'haystack.component.name=' in line:
                current_component['name'] = line.split('=')[1].strip()

            elif 'haystack.component.type=' in line:
                current_component['type'] = line.split('=')[1].strip()

            elif 'haystack.component.visits=' in line:
                current_component['n_visits'] = int(line.split('=')[1].strip())

            elif 'haystack.component.input=' in line:
                current_component['input_str'] = line.split('=', 1)[1].strip()

            elif 'haystack.component.output=' in line:
                current_component['output_str'] = line.split('=', 1)[1].strip()

    # Add the last component if exists
    if current_component:
        components.append(Component(**current_component))
    
    # Return components in reverse order to use as a stack
    return components[::-1]

def compare_traces(file1: str, file2: str) -> bool:
    stack1 = parse_log_file(file1)
    stack2 = parse_log_file(file2)
    
    # check if the number of components is the same
    if len(stack1) != len(stack2):
        print(f"Different number of components: {len(stack1)} vs {len(stack2)}")
        return False
    
    # compare components one by one
    for i in range(len(stack1)):
        comp1 = stack1[i]
        comp2 = stack2[i]
        
        if comp1.name != comp2.name or comp1.type != comp2.type:
            print(f"Mismatch between {comp1.name} and {comp2.name} in component name/type:")
            print(f"{file1}: {comp1.name} ({comp1.type})")
            print(f"{file2}: {comp2.name} ({comp2.type})")
            return False

        if comp1.input_str != comp2.input_str:
            print(f"Mismatch between {comp1.name} and {comp2.name} inputs:")
            print(f"{file1}: {comp1.input_str}")
            print(f"{file2}: {comp2.input_str}")
            return False

        if comp1.output_str != comp2.output_str:
            print(f"Mismatch between {comp1.name} and {comp2.name} outputs:")
            print(f"{file1}: {comp1.output_str}")
            print(f"{file2}: {comp2.output_str}")
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Compare component execution order between two trace files')
    parser.add_argument('file1', help='Path to the first trace file')
    parser.add_argument('file2', help='Path to the second trace file')
    
    args = parser.parse_args()
    
    are_identical = compare_traces(args.file1, args.file2)
    if are_identical:
        print("The execution order is identical in both traces")
    else:
        print("The execution order differs between traces")


if __name__ == "__main__":
    main()
