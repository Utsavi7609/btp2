import os

def list_files(startpath):
    res = []
    for root, dirs, files in os.walk(startpath):
        rel = os.path.relpath(root, startpath)
        level = rel.count(os.sep) if rel != '.' else 0
        indent = ' ' * 4 * level
        res.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            res.append(f"{subindent}{f}")
    return "\n".join(res)

if __name__ == "__main__":
    target = 'D:/BTP/btp2/'
    f_list = list_files(target)
    with open('btp_full_inventory.txt', 'w', encoding='utf-8') as f:
        f.write(f_list)
    print("✅ Full inventory saved to btp_full_inventory.txt")
