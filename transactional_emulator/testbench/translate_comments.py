#!/usr/bin/env python3
"""
Batch translate Chinese comments to English in Python files
"""

import re
from pathlib import Path

# Translation dictionary for common patterns
TRANSLATIONS = {
    # Common words
    "子矩阵": "sub-matrix",
    "矩阵": "matrix",
    "行引用": "row reference",
    "列引用": "column reference",
    "转置": "transpose",
    "代表": "represents",
    "用于": "used for",
    "获取": "get",
    "加载": "load",
    "注册": "register",
    "分配": "allocate",
    "释放": "free",
    "地址": "address",
    "偏移": "offset",
    "形状": "shape",
    "大小": "size",
    "权重": "weight",
    "激活": "activation",
    "批次": "batch",
    "结果": "result",
    "计算": "compute",
    "生成": "generate",
    "管理器": "manager",
    "符号表": "symbol table",
    "编译器": "compiler",
    "变量": "variable",
    "参数": "parameter",
    "返回": "return",
    "示例": "example",
    "注意": "note",
    "重要": "important",
    "警告": "warning",
    "前提条件": "prerequisites",
    "使用方法": "usage",
    "用法": "usage",
    "说明": "description",
    "功能": "feature",
    "属性": "property",
    "方法": "method",
    "函数": "function",
    "类": "class",
    "对象": "object",
    "接口": "interface",
    "实现": "implementation",
    "类型": "type",
    "名称": "name",
    "索引": "index",
    "块": "block",
    "子块": "sub-block",
    "行": "row",
    "列": "column",
    "行索引": "row index",
    "列索引": "column index",
    "起始": "start",
    "结束": "end",
    "基地址": "base address",
    "目标": "target",
    "源": "source",
    "输入": "input",
    "输出": "output",
    "中间": "intermediate",
    "临时": "temporary",
    "存储": "storage",
    "格式": "format",
    "布局": "layout",
    "分块": "block",
    "元信息": "metadata",
    "信息": "info",
    "缓存": "cache",
    "表": "table",
    "映射": "mapping",
    "字典": "dict",
    "列表": "list",
    "栈": "stack",
    "队列": "queue",
    "寄存器": "register",
    "指令": "instruction",
    "代码": "code",
    "生成的": "generated",
    "累积": "accumulated",
    "完整": "complete",
    "部分": "partial",
    "全部": "all",
    "单个": "single",
    "多个": "multiple",
    "第一": "first",
    "第二": "second",
    "第三": "third",
    "最后": "last",
    "当前": "current",
    "下一个": "next",
    "前一个": "previous",
    "开始": "start",
    "结束": "end",
    "初始化": "initialize",
    "重置": "reset",
    "清空": "clear",
    "更新": "update",
    "设置": "set",
    "添加": "add",
    "删除": "delete",
    "插入": "insert",
    "移除": "remove",
    "替换": "replace",
    "查找": "find",
    "查询": "query",
    "检查": "check",
    "验证": "validate",
    "确认": "confirm",
    "确定": "determine",
    "判断": "judge",
    "比较": "compare",
    "匹配": "match",
    "包含": "contain",
    "存在": "exist",
    "可用": "available",
    "有效": "valid",
    "无效": "invalid",
    "正确": "correct",
    "错误": "error",
    "成功": "success",
    "失败": "failure",
    "完成": "done",
    "未完成": "incomplete",
    "已加载": "loaded",
    "未加载": "not loaded",
    "已分配": "allocated",
    "未分配": "not allocated",
}

def main():
    print("Translation script - will only print patterns, not modify files yet")
    print("=" * 80)
    
    files = [
        "plena_program.py",
        "developer_compiler.py",
        "sub_matrix_manager.py",
        "symbol_table.py"
    ]
    
    for filename in files:
        filepath = Path(__file__).parent / filename
        if not filepath.exists():
            print(f"⚠️  File not found: {filepath}")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count Chinese characters in comments
        comment_pattern = r'#.*[\u4e00-\u9fff].*|"""[\s\S]*?[\u4e00-\u9fff][\s\S]*?"""|\'\'\'[\s\S]*?[\u4e00-\u9fff][\s\S]*?\'\'\''
        chinese_comments = re.findall(comment_pattern, content)
        
        if chinese_comments:
            print(f"\n{filename}: Found {len(chinese_comments)} comments with Chinese")
            # Show first few
            for i, comment in enumerate(chinese_comments[:3]):
                preview = comment[:100].replace('\n', ' ')
                print(f"  {i+1}. {preview}...")
        else:
            print(f"\n{filename}: ✓ No Chinese comments found")

if __name__ == "__main__":
    main()

