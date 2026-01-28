#!/usr/bin/env python3
"""
Test script for attention code generation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compiler.asm_templates.code_gen_op import _generate_vector_op

def test_attention_code_generation():
    """Test the attention code generation function"""
    
    # Test node with LLaMA-3.1 8B parameters
    hidden_size = 4096
    VLEN = 32
    op_config = {
        "name": "V_ADD_VV",
        "type": "vector",
        "reg_in_0": "i0",
        "reg_in_1": "i1",
        "reg_out": "i2",
        "loops": hidden_size // VLEN
    }
    
    # Generate the assembly code
    generated_code = _generate_vector_op(op_config)
    
    print("Generated Vector Assembly Code:")
    print("=" * 50)
    print(generated_code)
    print("=" * 50)
    
if __name__ == "__main__":
    test_attention_code_generation() 