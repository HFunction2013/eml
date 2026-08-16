from .eml import *
# 重新设计解析器
class ExpressionParser:
    """安全的表达式解析器"""
    
    def __init__(self):
        # 定义可用的函数和常量
        self.functions = {
            # 基本函数
            'EXP': EXP,
            'LN': LN,
            'LOG': LOG,
            'LOG2': LOG2,
            'LOG10': LOG10,
            
            # 算术运算
            'ADD': ADD,
            'SUB': SUB,
            'MUL': MUL,
            'DIV': DIV,
            'NEG': NEG,
            'POW': POW,
            'RECIP': RECIP,
            
            # 常用函数
            'SQRT': SQRT,
            'CBRT': CBRT,
            'ROOT_N': ROOT_N,
            'ABS': ABS,
            'SGN': SGN,
            'SQUARE': SQUARE,
            'CUBE': CUBE,
            
            # 三角函数
            'SIN': SIN,
            'COS': COS,
            'TAN': TAN,
            'COT': COT,
            'SEC': SEC,
            'CSC': CSC,
            
            # 反三角函数
            'ARCSIN': ARCSIN,
            'ARCCOS': ARCCOS,
            'ARCTAN': ARCTAN,
            'ARCCOT': ARCCOT,
            'ARCSEC': ARCSEC,
            'ARCCSC': ARCCSC,
            
            # 双曲函数
            'SINH': SINH,
            'COSH': COSH,
            'TANH': TANH,
            'COTH': COTH,
            'SECH': SECH,
            'CSCH': CSCH,
            
            # 反双曲函数
            'ARCSINH': ARCSINH,
            'ARCCOSH': ARCCOSH,
            'ARCTANH': ARCTANH,
            'ARCCOTH': ARCCOTH,
            'ARCSECH': ARCSECH,
            'ARCCSCH': ARCCSCH,
            
            # 组合函数
            'SINC': SINC,
            'SINHC': SINHC,
            'VERSINE': VERSINE,
            'HAVERSINE': HAVERSINE,
            'LOGISTIC': LOGISTIC,
            'HYPOT': HYPOT,
            'AVG': AVG,
            'MAX': MAX,
            'MIN': MIN,
            'CLAMP': CLAMP,
            'STEP': STEP,
            'RECT': RECT,
            'SIGNUM': SIGNUM,
            'RAMP': RAMP,
            'MANHATTAN': MANHATTAN,
            'CHEBYSHEV': CHEBYSHEV,
            
            # 简写
            'INV_SQRT': INV_SQRT,
            'INV_SQUARE': INV_SQUARE,
            'EXP2': EXP2,
            'EXP10': EXP10,
        }
        
        self.constants = {
            'X': X,
            'ONE': ONE,
            'E': E,
            'ZERO': ZERO,
            'TWO': TWO,
            'THREE': THREE,
            'NEG1': NEG1,
            'I': I,
            'PI': PI,
            'PHI': PHI,
        }
        
        # 运算符优先级
        self.precedence = {
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '^': 3,
            'u-': 4,  # 一元负号
        }
        
    def parse(self, expression: str) -> Node:
        """
        解析数学表达式字符串
        
        Args:
            expression: 表达式字符串，如 "SIN(2*PI*X)"
            
        Returns:
            Node: 表达式树
        """
        # 使用递归下降解析器
        self.tokens = self._lex(expression)
        self.pos = 0
        return self._parse_expression()
    
    def _lex(self, expression: str) -> list:
        """词法分析"""
        tokens = []
        i = 0
        n = len(expression)
        
        # 定义关键字模式
        keywords = set(self.functions.keys()) | set(self.constants.keys())
        
        while i < n:
            ch = expression[i]
            
            # 跳过空白
            if ch.isspace():
                i += 1
                continue
                
            # 数字
            if ch.isdigit() or ch == '.':
                j = i
                dot_seen = (ch == '.')
                while j < n and (expression[j].isdigit() or expression[j] == '.'):
                    if expression[j] == '.':
                        if dot_seen:
                            break
                        dot_seen = True
                    j += 1
                num_str = expression[i:j]
                try:
                    if '.' in num_str:
                        tokens.append(('NUM', float(num_str)))
                    else:
                        tokens.append(('NUM', int(num_str)))
                except ValueError:
                    raise ValueError(f"Invalid number: {num_str}")
                i = j
                continue
                
            # 标识符（函数、常量、变量）
            if ch.isalpha():
                j = i
                while j < n and (expression[j].isalnum() or expression[j] == '_'):
                    j += 1
                ident = expression[i:j]
                
                # 检查是否是关键字
                if ident.upper() in keywords:
                    ident = ident.upper()
                    if ident in self.functions:
                        tokens.append(('FUNC', ident))
                    elif ident in self.constants:
                        tokens.append(('CONST', ident))
                elif ident == 'X' or ident == 'x':
                    tokens.append(('VAR', 'X'))
                else:
                    # 尝试匹配大小写不敏感
                    ident_upper = ident.upper()
                    if ident_upper in keywords:
                        if ident_upper in self.functions:
                            tokens.append(('FUNC', ident_upper))
                        elif ident_upper in self.constants:
                            tokens.append(('CONST', ident_upper))
                    else:
                        # 尝试查找最长的匹配
                        matched = False
                        for k in range(len(ident), 0, -1):
                            test_ident = ident[:k].upper()
                            if test_ident in self.functions or test_ident in self.constants:
                                tokens.append(('FUNC' if test_ident in self.functions else 'CONST', test_ident))
                                i += k
                                matched = True
                                break
                        
                        if not matched:
                            raise ValueError(f"Unknown identifier: {ident}")
                        continue
                
                i = j
                continue
                
            # 运算符
            if ch in '+-*/^':
                tokens.append(('OP', ch))
                i += 1
                continue
                
            # 括号和逗号
            if ch == '(':
                tokens.append(('LPAREN', '('))
            elif ch == ')':
                tokens.append(('RPAREN', ')'))
            elif ch == ',':
                tokens.append(('COMMA', ','))
            else:
                raise ValueError(f"Invalid character: {ch}")
            i += 1
        
        tokens.append(('EOF', ''))
        return tokens
    
    def _current_token(self):
        """获取当前token"""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ('EOF', '')
    
    def _next_token(self):
        """获取下一个token"""
        self.pos += 1
        return self._current_token()
    
    def _parse_expression(self, precedence=0):
        """解析表达式"""
        # 解析一元表达式
        token_type, token_value = self._current_token()
        
        if token_type == 'OP' and token_value == '-':
            self._next_token()  # 跳过 '-'
            expr = NEG(self._parse_expression(self.precedence['u-']))
        elif token_type == 'OP' and token_value == '+':
            self._next_token()  # 跳过 '+'
            expr = self._parse_expression(self.precedence['u-'])
        elif token_type == 'LPAREN':
            self._next_token()  # 跳过 '('
            expr = self._parse_expression()
            if self._current_token()[0] != 'RPAREN':
                raise ValueError("Expected ')'")
            self._next_token()  # 跳过 ')'
        elif token_type == 'NUM':
            num = token_value
            self._next_token()
            if isinstance(num, int):
                expr = CONST(num)
            else:
                # 将浮点数转换为分数
                from fractions import Fraction
                frac = Fraction(num).limit_denominator(1000)
                if frac.denominator == 1:
                    expr = CONST(frac.numerator)
                else:
                    expr = DIV(CONST(frac.numerator), CONST(frac.denominator))
        elif token_type == 'VAR':
            self._next_token()
            expr = X()
        elif token_type == 'CONST':
            const_name = token_value
            self._next_token()
            if const_name in self.constants:
                expr = self.constants[const_name]()
            else:
                raise ValueError(f"Unknown constant: {const_name}")
        elif token_type == 'FUNC':
            func_name = token_value
            self._next_token()
            
            # 解析参数列表
            if self._current_token()[0] != 'LPAREN':
                raise ValueError(f"Function {func_name} expects '(' after it")
            self._next_token()  # 跳过 '('
            
            # 解析参数
            args = []
            if self._current_token()[0] != 'RPAREN':
                while True:
                    args.append(self._parse_expression())
                    if self._current_token()[0] != 'COMMA':
                        break
                    self._next_token()  # 跳过 ','
            
            if self._current_token()[0] != 'RPAREN':
                raise ValueError(f"Function {func_name} expects ')' after its parameter list")
            self._next_token()  # 跳过 ')'
            
            # 调用函数
            if func_name in self.functions:
                func = self.functions[func_name]
                expr = func(*args)
            else:
                raise ValueError(f"Unknown function: {func_name}")
        else:
            raise ValueError(f"Unexpected token: {token_type} '{token_value}'")
        
        # 解析二元运算符
        while True:
            token_type, token_value = self._current_token()
            if token_type != 'OP' or token_value not in '+-*/^':
                break
            
            if self.precedence.get(token_value, 0) <= precedence:
                break
                
            op = token_value
            self._next_token()
            
            right = self._parse_expression(self.precedence[op])
            
            if op == '+':
                expr = ADD(expr, right)
            elif op == '-':
                expr = SUB(expr, right)
            elif op == '*':
                expr = MUL(expr, right)
            elif op == '/':
                expr = DIV(expr, right)
            elif op == '^':
                expr = POW(expr, right)
        
        return expr
 
def parse_expr(expr_str: str) -> Node:
    """
    简化接口：解析数学表达式字符串
    
    Args:
        expr_str: 表达式字符串
        
    Returns:
        Node: 表达式树
    """
    parser = ExpressionParser()
    return parser.parse(expr_str)


def evaluate_expr(expr_str: str, x: complex) -> (Node, complex):
    """
    简化接口：解析并计算表达式
    
    Args:
        expr_str: 表达式字符串
        x: 变量值
        
    Returns:
        complex: 计算结果
    """
    expr = parse_expr(expr_str)
    return (expr, expr.evaluate(x))