from typing import Optional
import cmath
from .constants import NodeType

# 使用全局变量eml
eml = 'f'
# eml(x,y)=e^x-ln(y)

class Node:
    def __init__(
        self, 
        node_type: NodeType,
        left: Optional['Node'] = None, 
        right: Optional['Node'] = None
    ):
        self.node_type = node_type
        self.left = left
        self.right = right
    
    def __repr__(self) -> str:
        """返回节点的字符串表示"""
        if self.node_type == NodeType.X:
            return "X"
        elif self.node_type == NodeType.ONE:
            return "1"
        elif self.node_type == NodeType.EML:
            left_str = repr(self.left) if self.left else "_"
            right_str = repr(self.right) if self.right else "_"
            # 使用全局变量eml
            return f"{eml}({left_str}, {right_str})"
        return f"Unknown({self.node_type})"
    
    def __str__(self) -> str:
        """返回节点的字符串表示（同__repr__）"""
        return self.__repr__()
    
    def __add__(self, other: 'Node') -> 'Node':
        """重载+运算符: ADD(x,y) = x - (-y)"""
        return ADD(self, other)
    
    def __sub__(self, other: 'Node') -> 'Node':
        """重载-运算符: SUB(x,y) = EML(LN(x), e^y)"""
        return SUB(self, other)
    
    def __neg__(self) -> 'Node':
        """重载负号运算符: NEG(x) = SUB(0, x)"""
        return NEG(self)
    
    def __mul__(self, other: 'Node') -> 'Node':
        """重载*运算符: MUL(x,y) = e^(LN(x) + LN(y))"""
        return MUL(self, other)
    
    def __truediv__(self, other: 'Node') -> 'Node':
        """重载/运算符: DIV(x,y) = x * RECIP(y)"""
        return DIV(self, other)
    
    def __pow__(self, other: 'Node') -> 'Node':
        """重载**运算符: POW(x,y) = e^(y * ln(x))"""
        return POW(self, other)
        
    def evaluate(self, x: complex) -> complex:
        """
        计算表达式在给定x值时的结果（支持复数）
        自动清理浮点数微小精度误差
        
        Args:
            x: 变量X的值，可以是实数或复数
            
        Returns:
            complex: 表达式的计算结果（无微小误差）
        """
        def clean_complex(z: complex, tol: float = 1e-6, ndigits: int = 10) -> complex:
            """
            清理复数的微小误差：
            1. 绝对值 < tol 的部分 → 置为 0
            2. 对剩余部分进行四舍五入
            """
            real = z.real
            imag = z.imag
            
            # 清理微小值
            if abs(real) < tol:
                real = 0.0
            elif ndigits is not None:
                real = round(real, ndigits)
            
            if abs(imag) < tol:
                imag = 0.0
            elif ndigits is not None:
                imag = round(imag, ndigits)
            
            return complex(real, imag)

        if self.node_type == NodeType.X:
            return clean_complex(x)
        
        elif self.node_type == NodeType.ONE:
            return 1.0 + 0.0j
        
        elif self.node_type == NodeType.EML:
            if not self.left or not self.right:
                raise ValueError("EML节点必须有左右子节点")
            
            # 递归计算子节点
            left_val = self.left.evaluate(x)
            right_val = self.right.evaluate(x)
            
            # 计算指数与对数
            exp_val = cmath.exp(left_val)
            
            # 处理对数 0 情况
            if abs(right_val) < 1e-15:
                return complex(float('-1e6'), 0.0)
            ln_val = cmath.log(right_val)
            
            # 核心计算 + 自动清理误差
            result = exp_val - ln_val
            return clean_complex(result)
            
# 基本节点
ONE_NODE = Node(NodeType.ONE)
X_NODE = Node(NodeType.X)

# ==================== 基本函数 ====================
def EXP(x: Node) -> Node:
    """EXP(x) = f(x, 1)"""
    return Node(NodeType.EML, x, ONE_NODE)

def LN(x: Node) -> Node:
    """LN(x) = f(1, f(f(1, x), 1))"""
    inner_eml = Node(NodeType.EML, ONE_NODE, x)
    middle_eml = Node(NodeType.EML, inner_eml, ONE_NODE)
    return Node(NodeType.EML, ONE_NODE, middle_eml)

# ==================== 基本常量 ====================
def ONE() -> Node:
    """常数1"""
    return ONE_NODE

def X() -> Node:
    """变量X"""
    return X_NODE

def E() -> Node:
    """常数E = f(1, 1)"""
    return Node(NodeType.EML, ONE_NODE, ONE_NODE)

# ==================== 算术运算 ====================
def SUB(x: Node, y: Node) -> Node:
    """SUB(x, y) = f(LN(x), e^y)"""
    return Node(NodeType.EML, LN(x), EXP(y))

def NEG(x: Node) -> Node:
    """NEG(x) = SUB(0, x)"""
    return SUB(ZERO(), x)

def ZERO() -> Node:
    """ZERO = SUB(1, 1)"""
    return SUB(ONE(), ONE())

def ADD(x: Node, y: Node) -> Node:
    """ADD(x, y) = x - (-y)"""
    return SUB(x, NEG(y))

def MUL(x: Node, y: Node) -> Node:
    """MUL(x, y) = e^(LN(x) + LN(y))"""
    return EXP(ADD(LN(x), LN(y)))

def RECIP(x: Node) -> Node:
    """RECIP(x) = e^(-ln(x)) = 1/x"""
    return EXP(NEG(LN(x)))

def DIV(x: Node, y: Node) -> Node:
    """DIV(x, y) = x * RECIP(y)"""
    return MUL(x, RECIP(y))

def HALF(x: Node) -> Node:
    """HALF(x) = x/2"""
    return DIV(x, TWO())
    
def POW(x: Node, y: Node) -> Node:
    """POW(x, y) = e^(y * ln(x))"""
    return EXP(MUL(y, LN(x)))

# ==================== 常用常量 ====================
def NEG1() -> Node:
    """NEG1 = NEG(1) = -1"""
    return NEG(ONE())

def TWO() -> Node:
    """TWO = ADD(1, 1) = 2"""
    return ADD(ONE(), ONE())

def THREE() -> Node:
    """THREE = ADD(TWO, 1) = 3"""
    return ADD(TWO(), ONE())

def I() -> Node:
    """I = -SQRT(-1) = -√(-1) = i (主值)"""
    return NEG(SQRT(NEG1()))

# ==================== 简化函数 ====================
def CONST(n: int) -> Node:
    """构造任意整数常量 n"""
    if n == 0:
        return ZERO()
    elif n == 1:
        return ONE()
    elif n == 2:
        return TWO()
    elif n == 3:
        return THREE()
    
    if n > 0:
        # 使用幂运算表示大数
        if n == 10:
            return MUL(TWO(), CONST(5))  # 2 * 5
        elif n == 100:
            return POW(CONST(10), TWO())  # 10^2
        else:
            # 回退到二进制分解
            binary = bin(n)[2:]
            result = None
            for bit in binary:
                if result is not None:
                    result = ADD(result, result)  # 乘以2
                if bit == '1':
                    if result is None:
                        result = ONE()
                    else:
                        result = ADD(result, ONE())
            return result
    else:
        return NEG(CONST(-n))

def STEP(x: Node) -> Node:  # 阶跃函数
    """Heaviside阶跃函数"""
    return HALF(ADD(SGN(x), ONE()))

def RECT(x: Node, a: Node = HALF(ONE())) -> Node:  # 矩形函数
    """矩形函数"""
    return SUB(STEP(ADD(x, a)), STEP(SUB(x, a)))

def FRAC(p: Node, q: Node) -> Node:
    """有理数 p/q"""
    return DIV(p, q)

def SIGNUM(x: Node) -> Node:  # 三值符号函数
    """返回-1, 0, 1"""
    return STEP(x) - STEP(NEG(x))

def EXP2(x: Node) -> Node:  # 2^x
    """指数函数以2为底"""
    return POW(TWO(), x)
    
def LT(x: Node, y: Node) -> Node:
    """小于：x < y 返回1，否则0"""
    return STEP(SUB(y, x))

def LE(x: Node, y: Node) -> Node:
    """小于等于：x ≤ y 返回1，否则0"""
    return SUB(ONE(), STEP(SUB(x, y)))
    
def AND(x: Node, y: Node) -> Node:
    """逻辑与：min(x, y) 当x,y∈{0,1}"""
    return MIN(x, y)

def OR(x: Node, y: Node) -> Node:
    """逻辑或：max(x, y) 当x,y∈{0,1}"""
    return MAX(x, y)

def INV_SQRT(x: Node) -> Node:  # 1/√x
    """平方根倒数"""
    return RECIP(SQRT(x))

def INV_SQUARE(x: Node) -> Node:  # 1/x²
    """平方倒数"""
    return RECIP(SQUARE(x))

def EXP10(x: Node) -> Node:  # 10^x
    """指数函数以10为底"""
    return POW(CONST(10), x)

def RAMP(x: Node) -> Node:  # 斜坡函数
    """max(x, 0)"""
    return MAX(x, ZERO())

def MANHATTAN(x: Node, y: Node) -> Node:  # 曼哈顿距离
    """|x| + |y|"""
    return ADD(ABS(x), ABS(y))

def CHEBYSHEV(x: Node, y: Node) -> Node:  # 切比雪夫距离
    """max(|x|, |y|)"""
    return MAX(ABS(x), ABS(y))

def RATIONAL(p: int, q: int) -> Node:
    """有理数 p/q"""
    return FRAC(CONST(p), CONST(q))

def SQUARE(x: Node) -> Node:
    """SQUARE(x) = x²"""
    return MUL(x, x)

def CUBE(x: Node) -> Node:
    """CUBE(x) = x³"""
    return MUL(x, MUL(x, x))

def HYPOT(x: Node, y: Node) -> Node:
    """HYPOT(x, y) = √(x² + y²)"""
    return SQRT(ADD(SQUARE(x), SQUARE(y)))

def ABS(x: Node) -> Node:
    """ABS(x) = √(x²)"""
    return SQRT(SQUARE(x))

def SGN(x: Node) -> Node:
    """符号函数: sgn(x) = x/|x|"""
    return DIV(x, ABS(x))

def AVG(x: Node, y: Node) -> Node:
    """平均值: (x + y)/2"""
    return HALF(ADD(x, y))

def MAX(x: Node, y: Node) -> Node:
    """最大值: max(x, y) = (x + y + |x - y|)/2"""
    return HALF(ADD(ADD(x, y), ABS(SUB(x, y))))

def MIN(x: Node, y: Node) -> Node:
    """最小值: min(x, y) = (x + y - |x - y|)/2"""
    return HALF(SUB(ADD(x, y), ABS(SUB(x, y))))

def CLAMP(x: Node, a: Node, b: Node) -> Node:
    """裁剪: clamp(x, a, b) = min(max(x, a), b)"""
    return MIN(MAX(x, a), b)

def SQRT(x: Node) -> Node:
    """SQRT(x) = √x"""
    return ROOT_N(x, TWO())

def CBRT(x: Node) -> Node:
    """CBRT(x) = ³√x"""
    return ROOT_N(x, THREE())

def ROOT_N(x: Node, n: Node) -> Node:
    """ROOT_N(x, n) = x^(1/n)"""
    return POW(x, RECIP(n))

# ==================== 三角函数 ====================
def COS(x: Node) -> Node:
    """COS(x) = (e^(ix) + e^(-ix)) / 2"""
    i_expr = I()
    ix = MUL(i_expr, x)
    return DIV(ADD(EXP(ix), EXP(NEG(ix))), TWO())

def SIN(x: Node) -> Node:
    """SIN(x) = (e^(ix) - e^(-ix)) / (2i)"""
    i_expr = I()
    ix = MUL(i_expr, x)
    return DIV(SUB(EXP(ix), EXP(NEG(ix))), MUL(TWO(), i_expr))

def TAN(x: Node) -> Node:
    """TAN(x) = SIN(x) / COS(x)"""
    return DIV(SIN(x), COS(x))

def COT(x: Node) -> Node:
    """COT(x) = COS(x) / SIN(x)"""
    return DIV(COS(x), SIN(x))

def SEC(x: Node) -> Node:
    """SEC(x) = 1 / COS(x)"""
    return RECIP(COS(x))

def CSC(x: Node) -> Node:
    """CSC(x) = 1 / SIN(x)"""
    return RECIP(SIN(x))

# ==================== 反三角函数 ====================
def ARCCOS(x: Node) -> Node:
    """ARCCOS(x) = -i * ln(x + i√(1-x²))"""
    i_expr = I()
    sqrt_part = SQRT(SUB(ONE(), SQUARE(x)))
    return MUL(NEG(i_expr), LN(ADD(x, MUL(i_expr, sqrt_part))))

def ARCSIN(x: Node) -> Node:
    """ARCSIN(x) = -i * ln(ix + √(1-x²))"""
    i_expr = I()
    sqrt_part = SQRT(SUB(ONE(), SQUARE(x)))
    return MUL(NEG(i_expr), LN(ADD(MUL(i_expr, x), sqrt_part)))

def ARCTAN(x: Node) -> Node:
    """ARCTAN(x) = (i/2) * ln((i-x)/(i+x))"""
    i_expr = I()
    return MUL(DIV(i_expr, TWO()), LN(DIV(SUB(i_expr, x), ADD(i_expr, x))))
    
def ARCCOT(x: Node) -> Node:
    """反余切: arccot(x) = π/2 - arctan(x)"""
    return SUB(DIV(PI(), TWO()), ARCTAN(x))

def ARCSEC(x: Node) -> Node:
    """反正割: arcsec(x) = arccos(1/x)"""
    return ARCCOS(RECIP(x))

def ARCCSC(x: Node) -> Node:
    """反余割: arccsc(x) = arcsin(1/x)"""
    return ARCSIN(RECIP(x))

# ==================== 双曲函数 ====================
def COSH(x: Node) -> Node:
    """COSH(x) = (e^x + e^(-x)) / 2"""
    return DIV(ADD(EXP(x), EXP(NEG(x))), TWO())

def SINH(x: Node) -> Node:
    """SINH(x) = (e^x - e^(-x)) / 2"""
    return DIV(SUB(EXP(x), EXP(NEG(x))), TWO())

def TANH(x: Node) -> Node:
    """TANH(x) = SINH(x) / COSH(x)"""
    return DIV(SINH(x), COSH(x))

def COTH(x: Node) -> Node:
    """COTH(x) = COSH(x) / SINH(x)"""
    return DIV(COSH(x), SINH(x))

def SECH(x: Node) -> Node:
    """SECH(x) = 1 / COSH(x)"""
    return RECIP(COSH(x))

def CSCH(x: Node) -> Node:
    """CSCH(x) = 1 / SINH(x)"""
    return RECIP(SINH(x))

# ==================== 反双曲函数 ====================
def ARCSINH(x: Node) -> Node:
    """ARCSINH(x) = ln(x + √(x²+1))"""
    return LN(ADD(x, SQRT(ADD(SQUARE(x), ONE()))))

def ARCCOSH(x: Node) -> Node:
    """ARCCOSH(x) = ln(x + √(x²-1))"""
    return LN(ADD(x, SQRT(SUB(SQUARE(x), ONE()))))

def ARCTANH(x: Node) -> Node:
    """ARCTANH(x) = (1/2) * ln((1+x)/(1-x))"""
    return MUL(HALF(ONE()), LN(DIV(ADD(ONE(), x), SUB(ONE(), x))))
    
def ARCCOTH(x: Node) -> Node:
    """反双曲余切: arccoth(x) = (1/2) * ln((x+1)/(x-1)) for |x| > 1"""
    return MUL(HALF(ONE()), LN(DIV(ADD(x, ONE()), SUB(x, ONE()))))

def ARCSECH(x: Node) -> Node:
    """反双曲正割: arcsech(x) = ln((1 + √(1-x²))/x) for 0 < x ≤ 1"""
    sqrt_part = SQRT(SUB(ONE(), SQUARE(x)))
    return LN(DIV(ADD(ONE(), sqrt_part), x))

def ARCCSCH(x: Node) -> Node:
    """反双曲余割: arccsch(x) = ln((1/x) + √(1 + 1/x²))"""
    recip_x = RECIP(x)
    sqrt_part = SQRT(ADD(ONE(), SQUARE(recip_x)))
    return LN(ADD(recip_x, sqrt_part))

# ==================== 常用组合函数 ====================
def SINC(x: Node) -> Node:
    """sinc函数: sin(x)/x"""
    return DIV(SIN(x), x)

def SINHC(x: Node) -> Node:
    """sinhc函数: sinh(x)/x"""
    return DIV(SINH(x), x)

def VERSINE(x: Node) -> Node:
    """正矢函数: 1 - cos(x)"""
    return SUB(ONE(), COS(x))

def HAVERSINE(x: Node) -> Node:
    """半正矢函数: (1 - cos(x))/2"""
    return HALF(VERSINE(x))

def LOGISTIC(x: Node) -> Node:
    """逻辑函数: 1/(1 + e^(-x))"""
    return RECIP(ADD(ONE(), EXP(NEG(x))))

def LOG(base: Node, x: Node) -> Node:
    """对数: log_base(x) = ln(x)/ln(base)"""
    return DIV(LN(x), LN(base))

def LOG10(x: Node) -> Node:
    """常用对数: log10(x)"""
    return LOG(CONST(10), x)

def LOG2(x: Node) -> Node:
    """以2为底的对数: log2(x)"""
    return LOG(CONST(2), x)

# ==================== 数学常数 ====================
def PI() -> Node:
    """π = i * ln(-1)"""
    return MUL(I(), LN(NEG1()))

def PHI() -> Node:
    """黄金比例: (1 + √5)/2"""
    return HALF(ADD(ONE(), SQRT(CONST(5))))