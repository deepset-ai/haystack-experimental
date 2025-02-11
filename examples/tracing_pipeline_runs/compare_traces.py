import argparse
import ast
from dataclasses import dataclass
from typing import List


@dataclass
class Component:
    name: str
    type: str
    visits: int
    input_str: str
    output_str: str
    input_parsed: any
    output_parsed: any

def parse_structured_data(data_str: str) -> any:
    """
    Utility function to parse a string as Python literal.
    """
    try:
        # evaluate the string as a Python literal
        return ast.literal_eval(data_str)
    except (ValueError, SyntaxError):
        return data_str

def parse_log_file(file_path: str) -> List[Component]:
    """
    Reads haystack traces from a logfile.
    """
    components = []
    current_component = {}

    haystack_prefixes = {
        'name': 'haystack.component.name=',
        'type': 'haystack.component.type=',
        'visits': 'haystack.component.visits=',
        'input': 'haystack.component.input=',
        'output': 'haystack.component.output='
    }

    with open(file_path, 'r') as f:
        for line in f:

            if 'Operation: haystack.component.run' in line:
                if current_component:
                    # parse structured data before creating Component
                    current_component['input_parsed'] = parse_structured_data(current_component['input_str'])
                    current_component['output_parsed'] = parse_structured_data(current_component['output_str'])
                    components.append(Component(**current_component))
                    current_component = {}
                continue

            # parse component properties into current_component structure
            for key, prefix in haystack_prefixes.items():
                if prefix in line:
                    value = line.split('=', 1)[1].strip()
                    value = int(value) if key == 'visits' else value
                    if key == 'input':
                        current_component['input_str'] = value
                    elif key == 'output':
                        current_component['output_str'] = value
                    else:
                        current_component[key] = value
                    break

    # add the last component if exists
    if current_component:
        current_component['input_parsed'] = parse_structured_data(current_component['input_str'])
        current_component['output_parsed'] = parse_structured_data(current_component['output_str'])
        components.append(Component(**current_component))

    # return components in reverse order to use as a stack
    return components[::-1]

def compare_traces(file1: str, file2: str) -> bool:
    """
    Compare two trace files.

    Checks if the execution order, component names, types, inputs, outputs, and number of visits are the same.
    """
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
            print("")
            print(f"{file2}: {comp2.name} ({comp2.type})")
            return False

        if comp1.visits != comp2.visits:
            print(f"Mismatch between {comp1.name} and {comp2.name} in number of visits:")
            print(f"{file1}: {comp1.visits}")
            print("")
            print(f"{file2}: {comp2.visits}")
            return False

        if comp1.input_parsed != comp2.input_parsed:
            print(f"Mismatch between {comp1.name} and {comp2.name} inputs:")
            print(f"{file1}: {comp1.input_parsed}")
            print("")
            print(f"{file2}: {comp2.input_parsed}")
            return False

        if comp1.output_parsed != comp2.output_parsed:
            print(f"Mismatch between {comp1.name} and {comp2.name} outputs:")
            print(f"{file1}: {comp1.output_parsed}")
            print("")
            print(f"{file2}: {comp2.output_parsed}")
            return False
    
    return True

def main():
    """
    Main execution routine to compare pipeline traces.
    """
    parser = argparse.ArgumentParser(description='Compare component execution order between two trace files')
    parser.add_argument('file1', help='Path to the first trace file')
    parser.add_argument('file2', help='Path to the second trace file')
    
    args = parser.parse_args()
    
    are_identical = compare_traces(args.file1, args.file2)
    if are_identical:
        print("\nThe execution order is identical in both traces")
    else:
        print("\nThere are differences between traces")


if __name__ == "__main__":
    main()
