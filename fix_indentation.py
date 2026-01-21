#!/usr/bin/env python3
"""
Fix indentation issues in app.py
"""

def fix_app_indentation():
    """Fix the indentation issues in app.py"""
    
    # Read the file
    with open('app.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the problematic section and fix it
    fixed_lines = []
    in_evolution_section = False
    in_neural_section = False
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Detect the evolution section
        if 'if st.button("🧬 开始进化算法优化"' in line:
            in_evolution_section = True
            
        # Detect the neural section  
        if 'with evo_tabs[1]:' in line:
            in_neural_section = True
            in_evolution_section = False
            
        # Fix specific known issues
        if line_num == 3355 and line.strip().startswith('evolutionary_optimizer'):
            # Fix the indentation for line 3355
            fixed_lines.append('                    evolutionary_optimizer = get_evolutionary_optimizer()\n')
        elif line_num == 3360 and '# 配置进化算法' in line:
            # Fix the indentation for the spinner block
            fixed_lines.append('                                # 配置进化算法\n')
        else:
            fixed_lines.append(line)
    
    # Write the fixed file
    with open('app.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("Fixed indentation issues in app.py")

if __name__ == "__main__":
    fix_app_indentation()