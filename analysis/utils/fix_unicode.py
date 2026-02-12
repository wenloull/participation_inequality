
import os

files_to_fix = [
    r"c:/Users/dell/PycharmProjects/nlp2/participation_inequality/analysis/fig2.py",
    r"c:/Users/dell/PycharmProjects/nlp2/participation_inequality/analysis/intervention.py",
    r"c:/Users/dell/PycharmProjects/nlp2/participation_inequality/analysis/individualregression.py"
]

replacements = {
    '\u2705': '[OK]',
    '\u26a0': '[WARN]',
    '\u26a0\ufe0f': '[WARN]',
    '\u2713': '[OK]',
    '\u2714': '[OK]',
    '\U0001f4ca': '[GRAPH]',
    '✓': '[OK]',
    '⚠️': '[WARN]',
    '✅': '[OK]',
    '📊': '[GRAPH]'
}

for file_path in files_to_fix:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original_content = content
        for uni, rep in replacements.items():
            content = content.replace(uni, rep)
            
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed unicode in {file_path}")
        else:
            print(f"No unicode issues found in {file_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
