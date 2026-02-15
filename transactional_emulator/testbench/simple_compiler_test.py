"""
Simple Compiler Test: 从文本文件读取配置和伪代码，生成 ISA
只生成 ISA 代码，不运行模拟
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simple_compiler import SimpleCompiler


if __name__ == "__main__":
    # 创建简单编译器
    compiler = SimpleCompiler(real_data_ratio=1.125, mlen=64, blen=4)
    
    # 解析文件
    example_file = Path(__file__).parent / "example.txt"
    
    print("="*60)
    print("Simple Compiler Test - ISA Generation Only")
    print("="*60)
    print(f"\nParsing file: {example_file}\n")
    
    # 读取文件内容并显示
    with open(example_file, 'r') as f:
        file_content = f.read()
    print("File content:")
    print(file_content)
    print("\n" + "="*60)
    
    # 解析并生成代码
    try:
        code = compiler.parse_file(str(example_file))
        
        print("\n" + "="*60)
        print("Generated ISA Code:")
        print("="*60)
        # 显示前 2000 个字符
        if len(code) > 2000:
            print(code[:2000])
            print(f"\n... (total {len(code)} characters, {len(code.splitlines())} lines)")
        else:
            print(code)
        
        print("\n" + "="*60)
        print("Symbol Table:")
        print("="*60)
        compiler.print_symbol_table()
        
        # 保存生成的代码
        build_dir = Path(__file__).parent / "build"
        build_dir.mkdir(exist_ok=True)
        output_file = build_dir / "generated_asm_code.asm"
        with open(output_file, 'w') as f:
            f.write(code)
        
        print("\n" + "="*60)
        print("Summary:")
        print("="*60)
        print(f"✓ ISA code generated successfully")
        print(f"✓ Code saved to: {output_file}")
        print(f"✓ Total lines: {len(code.splitlines())}")
        print(f"✓ Total characters: {len(code)}")
        
        # 显示输入张量信息
        print(f"\nInput Tensors:")
        for name, tensor_info in compiler.input_tensors.items():
            print(f"  {name}: shape={tensor_info.shape}, hbm_addr={tensor_info.hbm_addr}, hbm_size={tensor_info.hbm_size}")
        
        # 显示结果张量信息
        symbol_table = compiler.get_symbol_table()
        result_tensors = []
        for alias, tensor_name in compiler.tensor_aliases.items():
            if alias in symbol_table:
                info = symbol_table[alias]
                if info.kind == "Batch" and info.hbm_addr == -1:  # result tensor
                    result_tensors.append((alias, info))
        
        if result_tensors:
            print(f"\nResult Tensors:")
            for alias, info in result_tensors:
                print(f"  {alias}: shape={info.shape}, vram_addr={info.vram_addr}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

