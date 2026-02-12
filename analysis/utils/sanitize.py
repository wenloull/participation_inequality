import re

def sanitize_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace non-ascii characters with their closest ascii equivalent or just remove them
    # For now, let's just replace them with a placeholder if they are not in comments, 
    # but simplest is to just replace common ones or strip.
    # Given the errors are in print statements, I'll replace them with '?' or similar.
    
    # specific replacements
    replacements = {
        '\u2713': '[OK]',
        '\u26a0': '[WARN]',
        '\u2714': '[OK]',
        '\U0001f4c8': '[GRAPH]',
        '\U0001f30d': '[WORLD]',
        '\u2022': '-',
        '\U0001f537': '[INFO]',
        '\U0001f4be': '[SAVED]',
        '\U0001f4ca': '[STATS]',
        '\U0001f50d': '[SEARCH]',
        '\U0001f4dd': '[NOTE]',
        '\ufe0f': '', # Variation selector
    }

    new_content = content
    for k, v in replacements.items():
        new_content = new_content.replace(k, v)
        
    # Generic replacement: Strip all other non-ascii characters
    # safe_content = re.sub(r'[^\x00-\x7F]', '', new_content)
    # Using a list comprehension to be explicit and safe
    safe_content = "".join([c if ord(c) < 128 else "" for c in new_content])
        
    if safe_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(safe_content)
        print(f"Sanitized {filepath} (removed non-ascii characters)")
    else:
        print(f"No changes for {filepath}")

if __name__ == "__main__":
    sanitize_file('c:/Users/dell/PycharmProjects/nlp2/participation_inequality/analysis/intervention.py')
    # sanitize_file('c:/Users/dell/PycharmProjects/nlp2/participation_inequality/analysis/figure3_rq2.py')
