from eml import evaluate_expr, parse_expr

def split_with_expression(input_str):
    """
    用词法分析的方式分割表达式和 with 子句
    支持: 表达式 with x=表达式
    """
    s = input_str.strip()
    
    # 查找 "with" (不区分大小写)
    s_lower = s.lower()
    with_pos = s_lower.find(" with ")
    
    if with_pos == -1:
        return s, None  # 没有 with 子句
    
    # 分割字符串
    expr_str = s[:with_pos].strip()
    rest = s[with_pos + 5:].strip()  # 5 是 " with" 的长度
    
    if not rest:
        return expr_str, None
    
    # 检查是否是 x= 格式
    if rest[0].lower() == 'x' and len(rest) > 1 and rest[1] == '=':
        # 找到 = 号的位置
        eq_pos = 1
    elif rest.lower().startswith("x ") and "=" in rest[1:]:
        # 处理 x = 的情况
        eq_pos = rest.find("=")
    else:
        return expr_str, None
    
    # 提取 = 后面的部分
    value_expr = rest[eq_pos+1:].strip()
    if not value_expr:
        return expr_str, None
    
    return expr_str, value_expr

while True:
    try:
        s = input("> ").strip()
        
        if s.lower() in ("exit", "quit"):
            print("Program Exited.")
            break
            
        if not s:
            continue
            
        if s.endswith("\\c"):
            s = s[:-2].strip()
            continue
        
        # 用词法分析检查是否包含 with
        expr_str, value_expr = split_with_expression(s)
        
        if value_expr is not None:
            # 有 with 子句
            try:
                # 计算右侧表达式的值 (用 x=0)
                _, x_value = evaluate_expr(value_expr, complex(0))
                
                # 计算主表达式
                expr_node, result = evaluate_expr(expr_str, x_value)
                print(f"x = {x_value}")
                print(f"Parsed:{expr_node}")
                print(f"Result: {result}")
                
            except Exception as e:
                print(f"Error: {e}")
        else:
            # 没有 with 子句，只解析
            expr_node = parse_expr(s)
            print(f"Parsed: {expr_node}")
        
    except Exception as e:
        print(f"Error: {e}")