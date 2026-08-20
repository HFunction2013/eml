"""
EML Reverse Compiler —— 从 EML 表达式 (f(a,b) = e^a - ln(b)) 逆向化简回原始数学表达式。

核心思路:
1. 解析 f(a,b) 格式的 EML 表达式为原始 AST (仅含 EML/X/1 节点)
2. 自底向上模式匹配, 逐层提升到高级运算 (exp/ln → sub → neg/add → mul/div/pow → 三角函数...)
3. 代数化简 (常量折叠、恒等律、exp/ln 互消、对数/指数法则)
4. 迭代到不动点, 输出标准数学记号

用法:
    from reverse_eml import reverse_eml
    print(reverse_eml("f(X, 1)"))           # => exp(x)
    print(reverse_eml("f(1, f(f(1, X), 1))"))  # => ln(x)
"""

from __future__ import annotations
from typing import Optional
import math
import cmath

# =====================================================================
# 1. 简化 AST
# =====================================================================

class SNode:
    """简化后的表达式节点。type 为运算类型, children 为子节点, value 为数值常量值。"""
    __slots__ = ('type', 'children', 'value')

    def __init__(self, type: str, *children, value=None):
        self.type = type
        self.children = list(children)
        self.value = value

    def __repr__(self):
        if self.type == 'num':
            return f"Num({self.value})"
        if self.type == 'var':
            return "Var(x)"
        args = ', '.join(repr(c) for c in self.children)
        return f"{self.type}({args})"

    def __eq__(self, other):
        """结构相等比较 (用于模式匹配)"""
        if not isinstance(other, SNode):
            return False
        if self.type != other.type:
            return False
        if self.type == 'num':
            return self.value == other.value
        if self.type == 'var':
            return True
        if len(self.children) != len(other.children):
            return False
        return all(a == b for a, b in zip(self.children, other.children))

    def __hash__(self):
        if self.type == 'num':
            return hash(('num', self.value))
        if self.type == 'var':
            return hash('var')
        return hash((self.type, tuple(hash(c) for c in self.children)))

    def copy(self):
        return SNode(self.type, *[c.copy() for c in self.children], value=self.value)


# 便捷构造函数
def Num(v): return SNode('num', value=v)
def Var(): return SNode('var')
def Add(a, b): return SNode('add', a, b)
def Sub(a, b): return SNode('sub', a, b)
def Mul(a, b): return SNode('mul', a, b)
def Div(a, b): return SNode('div', a, b)
def Neg(a): return SNode('neg', a)
def Pow(a, b): return SNode('pow', a, b)
def Exp(a): return SNode('exp', a)
def Ln(a): return SNode('ln', a)
def Sin(a): return SNode('sin', a)
def Cos(a): return SNode('cos', a)
def Tan(a): return SNode('tan', a)
def Sinh(a): return SNode('sinh', a)
def Cosh(a): return SNode('cosh', a)
def Tanh(a): return SNode('tanh', a)
def Sqrt(a): return SNode('sqrt', a)
def Abs(a): return SNode('abs', a)
def Eml(a, b): return SNode('eml', a, b)  # 兜底: 无法化简的 EML

ZERO = Num(0)
ONE = Num(1)
TWO = Num(2)
NEG_ONE = Num(-1)
E_CONST = SNode('const_e')
PI_CONST = SNode('const_pi')
I_CONST = SNode('const_i')


# =====================================================================
# 2. EML 表达式解析器 (f(a,b) 格式)
# =====================================================================

class EMLParser:
    """解析 f(a,b) 格式的 EML 表达式字符串为原始 AST"""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def parse(self):
        node = self._parse_expr()
        self._skip_ws()
        if self.pos < len(self.text):
            raise ValueError(f"Unexpected character at position {self.pos}: {self.text[self.pos]}")
        return node

    def _skip_ws(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _peek(self):
        self._skip_ws()
        return self.text[self.pos] if self.pos < len(self.text) else ''

    def _parse_expr(self):
        self._skip_ws()
        if self.pos >= len(self.text):
            raise ValueError("Unexpected end of expression")

        ch = self.text[self.pos]

        if ch == 'f' or ch == 'F':
            # f(...) 或 F(...)
            self.pos += 1
            self._skip_ws()
            if self.pos >= len(self.text) or self.text[self.pos] != '(':
                raise ValueError(f"Expected '(' after 'f' at position {self.pos}")
            self.pos += 1  # skip '('
            left = self._parse_expr()
            self._skip_ws()
            if self.pos >= len(self.text) or self.text[self.pos] != ',':
                raise ValueError(f"Expected ',' at position {self.pos}")
            self.pos += 1  # skip ','
            right = self._parse_expr()
            self._skip_ws()
            if self.pos >= len(self.text) or self.text[self.pos] != ')':
                raise ValueError(f"Expected ')' at position {self.pos}")
            self.pos += 1  # skip ')'
            return Eml(left, right)

        elif ch == 'X' or ch == 'x':
            self.pos += 1
            return Var()

        elif ch == '1':
            self.pos += 1
            return ONE

        elif ch.isdigit() or ch == '.':
            # 数字 (虽然 EML 规范只有 1, 但容错)
            start = self.pos
            while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
                self.pos += 1
            num_str = self.text[start:self.pos]
            val = float(num_str) if '.' in num_str else int(num_str)
            return Num(val)

        else:
            raise ValueError(f"Unexpected character '{ch}' at position {self.pos}")


def parse_eml(text: str) -> SNode:
    """解析 EML 表达式字符串"""
    return EMLParser(text).parse()


# =====================================================================
# 3. 从仓库 Node 对象转换 (可选, 方便测试)
# =====================================================================

def node_to_snode(node) -> SNode:
    """将仓库的 Node 对象转换为 SNode (EML 原始形式)"""
    # 延迟导入避免循环依赖
    nt = type(node).__name__
    # 通过 node_type 判断
    if hasattr(node, 'node_type'):
        ntype = node.node_type
        # NodeType.X=0, ONE=1, EML=2
        if ntype.name == 'X':
            return Var()
        elif ntype.name == 'ONE':
            return ONE
        elif ntype.name == 'EML':
            return Eml(node_to_snode(node.left), node_to_snode(node.right))
    raise ValueError(f"Unknown node type: {node}")


# =====================================================================
# 4. 模式提升 (EML → 高级运算)
# =====================================================================

def is_num(node, val=None):
    if node.type != 'num':
        return False
    return val is None or node.value == val


def _is_half(node):
    """判断是否为 1/2 (各种等价形式)"""
    if is_num(node, 0.5):
        return True
    if node.type == 'div' and is_num(node.children[0], 1) and is_num(node.children[1], 2):
        return True
    return False


def lift(node: SNode) -> SNode:
    """
    自底向上提升 EML 树到高级运算。
    关键: 复合模式 (如 LN) 在递归之前先匹配, 避免子节点被原子规则破坏。
    """
    # 叶子节点
    if node.type == 'var':
        return Var()
    if node.type == 'num':
        return Num(node.value)
    if node.type in ('const_e', 'const_pi', 'const_i'):
        return SNode(node.type)

    # 非 EML 节点 (已经是高级运算的子节点)
    if node.type != 'eml':
        lifted_children = [lift(c) for c in node.children]
        return SNode(node.type, *lifted_children, value=node.value)

    # node 是 EML(left, right)
    left_raw = node.children[0]
    right_raw = node.children[1]

    # ---- 复合模式优先匹配 (在递归之前) ----

    # LN(A) = f(1, f(f(1, A), 1))
    result = _try_lift_ln(left_raw, right_raw)
    if result is not None:
        return result

    # ---- 递归提升子节点 ----
    left = lift(left_raw)
    right = lift(right_raw)

    # ---- 原子模式匹配 (子节点已提升) ----

    # EXP(A) = f(A, 1)
    if is_num(right, 1):
        return Exp(left)

    # SUB(A, B) = f(LN(A), EXP(B))
    # 即 left 是 ln(A), right 是 exp(B)
    if left.type == 'ln' and right.type == 'exp':
        return Sub(left.children[0], right.children[0])

    # 兜底: EML(A, B) = exp(A) - ln(B)
    return Sub(Exp(left), Ln(right))


def _try_lift_ln(left_raw, right_raw):
    """
    匹配 LN 模式: f(1, f(f(1, A), 1)) → ln(A)
    必须在递归前匹配, 因为 f(f(1,A),1) 会被原子规则提升为 exp(f(1,A))
    """
    # left 必须是 1
    if not (left_raw.type == 'num' and left_raw.value == 1):
        return None

    # right 必须是 EML(inner, 1)
    if right_raw.type != 'eml':
        return None
    right_left = right_raw.children[0]
    right_right = right_raw.children[1]
    if not (right_right.type == 'num' and right_right.value == 1):
        return None

    # right_left 必须是 EML(1, A)
    if right_left.type != 'eml':
        return None
    rl_left = right_left.children[0]
    rl_right = right_left.children[1]
    if not (rl_left.type == 'num' and rl_left.value == 1):
        return None

    # 匹配成功! A = rl_right
    A = lift(rl_right)
    return Ln(A)


# =====================================================================
# 5. 代数化简
# =====================================================================

def simplify(node: SNode) -> SNode:
    """递归代数化简, 迭代到不动点"""
    prev = None
    curr = node
    for _ in range(50):  # 防止无限循环
        curr = _simplify_once(curr)
        if prev is not None and _structural_equal(prev, curr):
            break
        prev = curr
    return curr


def _structural_equal(a, b):
    if a.type != b.type:
        return False
    if a.type == 'num':
        return a.value == b.value
    if len(a.children) != len(b.children):
        return False
    return all(_structural_equal(x, y) for x, y in zip(a.children, b.children))


def _simplify_once(node: SNode) -> SNode:
    """单轮化简: 先化简子节点, 再化简当前节点"""
    # 叶子
    if node.type in ('var', 'num', 'const_e', 'const_pi', 'const_i'):
        return node

    # 先化简子节点
    children = [_simplify_once(c) for c in node.children]
    node = SNode(node.type, *children, value=node.value)

    # 按类型化简
    t = node.type
    if t == 'neg':
        return _simplify_neg(node)
    elif t == 'add':
        return _simplify_add(node)
    elif t == 'sub':
        return _simplify_sub(node)
    elif t == 'mul':
        return _simplify_mul(node)
    elif t == 'div':
        return _simplify_div(node)
    elif t == 'pow':
        return _simplify_pow(node)
    elif t == 'exp':
        return _simplify_exp(node)
    elif t == 'ln':
        return _simplify_ln(node)
    elif t == 'sqrt':
        return _simplify_sqrt(node)
    elif t in _TRIG_TYPES:
        return _simplify_trig(node)
    elif t == 'eml':
        # 兜底 EML: 提升为 exp(left) - ln(right) 再化简
        return _simplify_once(Sub(Exp(node.children[0]), Ln(node.children[1])))
    return node


def _simplify_neg(node):
    a = node.children[0]
    # -0 = 0
    if is_num(a, 0):
        return ZERO
    # -(-x) = x
    if a.type == 'neg':
        return a.children[0]
    # -(常数) = 常数
    if a.type == 'num':
        return Num(-a.value)
    # -(a - b) = b - a
    if a.type == 'sub':
        return Sub(a.children[1], a.children[0])
    # -(a + b) = -a + (-b)  (展开以便 exp/ln 化简)
    if a.type == 'add':
        return Add(Neg(a.children[0]), Neg(a.children[1]))
    # -(a * b)  where one is constant: (-c) * rest
    if a.type == 'mul':
        for i, child in enumerate(a.children):
            if child.type == 'num':
                rest = a.children[1 - i]
                return Mul(Num(-child.value), rest)
    return node


def _simplify_add(node):
    a, b = node.children[0], node.children[1]
    # a + 0 = a
    if is_num(a, 0):
        return b
    if is_num(b, 0):
        return a
    # 常数 + 常数
    if a.type == 'num' and b.type == 'num':
        return Num(a.value + b.value)
    # a + (-b) = a - b
    if b.type == 'neg':
        return Sub(a, b.children[0])
    # (-a) + b = b - a
    if a.type == 'neg':
        return Sub(b, a.children[0])
    # a + a = 2a
    if _structural_equal(a, b):
        return Mul(TWO, a)
    # ln(a) + ln(b) = ln(a*b) 当 a,b 为常量时
    if a.type == 'ln' and b.type == 'ln' and a.children[0].type == 'num' and b.children[0].type == 'num':
        product = a.children[0].value * b.children[0].value
        if isinstance(product, float) and product.is_integer():
            product = int(product)
        return Ln(Num(product))
    # 合并同类项: c1*x + c2*x = (c1+c2)*x (简单形式)
    if a.type == 'mul' and b.type == 'mul':
        ca, xa = a.children[0], a.children[1]
        cb, xb = b.children[0], b.children[1]
        if ca.type == 'num' and cb.type == 'num' and _structural_equal(xa, xb):
            return Mul(Num(ca.value + cb.value), xa)
    # 三角恒等式: sin²+cos²=1, 1+tan²=sec², 1+cot²=csc²
    result = _simplify_trig_identities_add(node)
    if result is not None:
        return result
    return node


def _simplify_sub(node):
    a, b = node.children[0], node.children[1]
    # a - 0 = a
    if is_num(b, 0):
        return a
    # 0 - a = -a
    if is_num(a, 0):
        return Neg(b)
    # a - a = 0
    if _structural_equal(a, b):
        return ZERO
    # 常数 - 常数
    if a.type == 'num' and b.type == 'num':
        return Num(a.value - b.value)
    # a - (-b) = a + b
    if b.type == 'neg':
        return Add(a, b.children[0])
    # a - b  where a and b are both ln: ln(x) - ln(y) = ln(x/y)
    if a.type == 'ln' and b.type == 'ln':
        return Ln(Div(a.children[0], b.children[0]))
    return node


def _simplify_mul(node):
    a, b = node.children[0], node.children[1]
    # a * 0 = 0
    if is_num(a, 0) or is_num(b, 0):
        return ZERO
    # a * 1 = a
    if is_num(a, 1):
        return b
    if is_num(b, 1):
        return a
    # a * (-1) = -a
    if is_num(a, -1):
        return Neg(b)
    if is_num(b, -1):
        return Neg(a)
    # 常数 * 常数
    if a.type == 'num' and b.type == 'num':
        return Num(a.value * b.value)
    # (-a) * b = -(a*b)
    if a.type == 'neg':
        return Neg(Mul(a.children[0], b))
    if b.type == 'neg':
        return Neg(Mul(a, b.children[0]))
    # a * a = a^2
    if _structural_equal(a, b):
        return Pow(a, TWO)
    # a^b * a = a^(b+1)
    if a.type == 'pow' and _structural_equal(a.children[0], b):
        return Pow(a.children[0], Add(a.children[1], ONE))
    if b.type == 'pow' and _structural_equal(b.children[0], a):
        return Pow(b.children[0], Add(b.children[1], ONE))
    # a^b * a^c = a^(b+c)
    if a.type == 'pow' and b.type == 'pow' and _structural_equal(a.children[0], b.children[0]):
        return Pow(a.children[0], Add(a.children[1], b.children[1]))
    # 注意: 不使用 exp(a)*exp(b)=exp(a+b), 避免与 exp 展开规则循环
    # a * (1/b) = a/b
    if b.type == 'div' and is_num(b.children[0], 1):
        return Div(a, b.children[1])
    if a.type == 'div' and is_num(a.children[0], 1):
        return Div(b, a.children[1])
    # a * recip(b) pattern: a * exp(-ln(b)) = a/b
    if b.type == 'exp' and b.children[0].type == 'neg' and b.children[0].children[0].type == 'ln':
        return Div(a, b.children[0].children[0].children[0])
    if a.type == 'exp' and a.children[0].type == 'neg' and a.children[0].children[0].type == 'ln':
        return Div(b, a.children[0].children[0].children[0])
    # 常量前置: x*2 → 2*x
    if a.type != 'num' and b.type == 'num':
        return Mul(b, a)
    # 三角恒等式: tan*cot=1, sec*cos=1, csc*sin=1
    result = _simplify_trig_identities_mul(node)
    if result is not None:
        return result
    return node


def _simplify_div(node):
    a, b = node.children[0], node.children[1]
    # 0 / a = 0
    if is_num(a, 0):
        return ZERO
    # a / 1 = a
    if is_num(b, 1):
        return a
    # a / a = 1
    if _structural_equal(a, b):
        return ONE
    # 常数 / 常数 — 仅在结果为整数时折叠, 保留有理数形式
    if a.type == 'num' and b.type == 'num' and b.value != 0:
        result = a.value / b.value
        if isinstance(result, float) and result.is_integer():
            return Num(int(result))
        # 非整数结果保留为除法 (不折叠为浮点数)
        return node
    # (a/b)/c = a/(b*c)  扁平化嵌套除法
    if a.type == 'div':
        return Div(a.children[0], Mul(a.children[1], b))
    # a/(b/c) = a*c/b
    if b.type == 'div':
        return Div(Mul(a, b.children[1]), b.children[0])
    # (a*c)/c = a  (约去公共因子)
    if a.type == 'mul':
        for i, child in enumerate(a.children):
            if _structural_equal(child, b):
                rest = a.children[1 - i]
                return rest
    # a/(a*c) = 1/c
    if b.type == 'mul':
        for i, child in enumerate(b.children):
            if _structural_equal(child, a):
                rest = b.children[1 - i]
                return Div(ONE, rest)
    # a / (-b) = -(a/b)
    if b.type == 'neg':
        return Neg(Div(a, b.children[0]))
    # 1 / exp(a) = exp(-a)
    if is_num(a, 1) and b.type == 'exp':
        return Exp(Neg(b.children[0]))
    # exp(a) / exp(b) = exp(a-b)
    if a.type == 'exp' and b.type == 'exp':
        return Exp(Sub(a.children[0], b.children[0]))
    return node


def _simplify_pow(node):
    a, b = node.children[0], node.children[1]
    # a^0 = 1
    if is_num(b, 0):
        return ONE
    # a^1 = a
    if is_num(b, 1):
        return a
    # 0^a = 0 (a>0)
    if is_num(a, 0) and b.type == 'num' and b.value > 0:
        return ZERO
    # 1^a = 1
    if is_num(a, 1):
        return ONE
    # 常数 ^ 常数 — 仅在结果为整数时折叠, 保留符号形式 (如 sqrt(5) 不展开为浮点数)
    if a.type == 'num' and b.type == 'num':
        # (-1)^0.5 或 (-1)^(1/2) → i (虚数单位主值)
        if a.value == -1 and b.value in (0.5, 1/2):
            return I_CONST
        try:
            result = a.value ** b.value
            if isinstance(result, complex):
                return node
            if isinstance(result, float) and result.is_integer():
                return Num(int(result))
            # 非整数结果保留为 pow (不折叠为浮点数)
            return node
        except (OverflowError, ValueError):
            return node
    # (a^b)^c = a^(b*c)
    if a.type == 'pow':
        # 特殊: (a^2)^(1/2) = |a| (不要化简为 a, 保留绝对值结构)
        if is_num(a.children[1], 2) and _is_half(b):
            return Abs(a.children[0])
        return Pow(a.children[0], Mul(a.children[1], b))
    # exp(a)^b = exp(a*b)
    if a.type == 'exp':
        return Exp(Mul(a.children[0], b))
    # a^(1/2) = sqrt(a)
    if b.type == 'div' and is_num(b.children[0], 1) and is_num(b.children[1], 2):
        return Sqrt(a)
    return node


def _expand_exp_add(add_node):
    """
    展开 exp(a + b + ...): 将 ln(x) 项提升为 x, -ln(x) 项提升为 1/x,
    其余项保留为 exp(term), 最终相乘。
    支持任意数量的加法项, 保持从左到右的顺序。
    """
    # 收集所有加法项 (保持顺序)
    terms = []
    def _collect(n):
        if n.type == 'add':
            _collect(n.children[0])
            _collect(n.children[1])
        else:
            terms.append(n)
    _collect(add_node)

    # 分类处理每一项
    result = ONE
    for term in terms:
        if term.type == 'ln':
            # exp(ln(x)) = x
            result = Mul(result, term.children[0])
        elif term.type == 'neg' and term.children[0].type == 'ln':
            # exp(-ln(x)) = 1/x
            result = Div(result, term.children[0].children[0])
        else:
            result = Mul(result, Exp(term))
    return result


def _simplify_exp(node):
    a = node.children[0]
    # exp(0) = 1
    if is_num(a, 0):
        return ONE
    # exp(1) = e
    if is_num(a, 1):
        return E_CONST
    # 常数
    if a.type == 'num':
        try:
            v = math.exp(a.value)
            if v.is_integer():
                return Num(int(v))
            return Num(v)
        except OverflowError:
            return node
    # exp(ln(x)) = x
    if a.type == 'ln':
        return a.children[0]
    # exp(a + b) 展开处理: 识别 ln(x) 和 -ln(x) 项
    if a.type == 'add':
        return _expand_exp_add(a)

    # exp(a - b) = exp(a) / exp(b)
    if a.type == 'sub':
        ax, bx = a.children[0], a.children[1]
        return Div(Exp(ax), Exp(bx))
    # exp(-ln(x)) = 1/x
    if a.type == 'neg' and a.children[0].type == 'ln':
        return Div(ONE, a.children[0].children[0])
    # exp(b * ln(a)) = a^b  (POW 的逆识别)
    if a.type == 'mul':
        # 找到 ln 因子
        ln_part = None
        other = None
        for child in a.children:
            if child.type == 'ln':
                ln_part = child
            else:
                other = child
        if ln_part is not None and other is not None:
            return Pow(ln_part.children[0], other)
        # b * ln(a) where b is first
        if a.children[1].type == 'ln':
            return Pow(a.children[1].children[0], a.children[0])
    # exp(ln(a) / b) = a^(1/b)
    if a.type == 'div' and a.children[0].type == 'ln':
        return Pow(a.children[0].children[0], Div(ONE, a.children[1]))
    # exp(-ln(a) / b) = a^(-1/b)
    if a.type == 'div' and a.children[0].type == 'neg' and a.children[0].children[0].type == 'ln':
        return Pow(a.children[0].children[0].children[0], Neg(Div(ONE, a.children[1])))
    # exp(2*ln(x)/2) = sqrt(x^2) = |x|  (ABS 模式)
    if a.type == 'div' and is_num(a.children[1], 2):
        inner = a.children[0]
        if inner.type == 'mul':
            if is_num(inner.children[0], 2) and inner.children[1].type == 'ln':
                return Abs(inner.children[1].children[0])
            if inner.children[0].type == 'ln' and is_num(inner.children[1], 2):
                return Abs(inner.children[0].children[0])
        if inner.type == 'add' and len(inner.children) == 2:
            c1, c2 = inner.children[0], inner.children[1]
            if c1.type == 'ln' and c2.type == 'ln' and _structural_equal(c1, c2):
                return Abs(c1.children[0])
    return node


def _simplify_ln(node):
    a = node.children[0]
    # ln(1) = 0
    if is_num(a, 1):
        return ZERO
    # ln(e) = 1
    if a.type == 'const_e':
        return ONE
    # ln(exp(x)) = x
    if a.type == 'exp':
        return a.children[0]
    # ln(a * b) = ln(a) + ln(b)
    if a.type == 'mul':
        return Add(Ln(a.children[0]), Ln(a.children[1]))
    # ln(a / b) = ln(a) - ln(b)
    if a.type == 'div':
        return Sub(Ln(a.children[0]), Ln(a.children[1]))
    # ln(a^b) = b * ln(a)
    if a.type == 'pow':
        return Mul(a.children[1], Ln(a.children[0]))
    # ln(1/x) = -ln(x)
    if a.type == 'div' and is_num(a.children[0], 1):
        return Neg(Ln(a.children[1]))
    # 常数 — 仅在结果为整数时折叠 (如 ln(1)=0, ln(e)=1), 保留 ln(整数) 符号形式
    if a.type == 'num' and a.value > 0:
        v = math.log(a.value)
        if abs(v - round(v)) < 1e-10:
            return Num(round(v))
        # 非整数结果保留为 ln (不折叠为浮点数)
        return node
    return node


def _simplify_sqrt(node):
    a = node.children[0]
    # sqrt(0) = 0
    if is_num(a, 0):
        return ZERO
    # sqrt(1) = 1
    if is_num(a, 1):
        return ONE
    # sqrt(-1) = i
    if is_num(a, -1):
        return I_CONST
    # sqrt(a^2) = |a|  (简化为 a, 假设主值)
    if a.type == 'pow' and is_num(a.children[1], 2):
        return Abs(a.children[0])
    # sqrt(exp(a)) = exp(a/2)
    if a.type == 'exp':
        return Exp(Div(a.children[0], TWO))
    # 常数 — 仅完全平方数折叠, 保留 sqrt(非平方数) 符号形式
    if a.type == 'num' and a.value >= 0:
        v = math.sqrt(a.value)
        if v.is_integer():
            return Num(int(v))
        # 非完全平方数保留为 sqrt
        return node
    return node


# ═══════════════════════════════════════════════════════════
#  三角函数化简 (奇偶性 / 诱导公式 / 周期性)
# ═══════════════════════════════════════════════════════════

# 奇函数: f(-x) = -f(x)
_ODD_TRIG = {'sin', 'tan', 'cot', 'csc', 'sinh', 'tanh', 'coth', 'csch'}
# 偶函数: f(-x) = f(x)
_EVEN_TRIG = {'cos', 'sec', 'cosh', 'sech'}
# 所有三角函数类型
_TRIG_TYPES = _ODD_TRIG | _EVEN_TRIG | {'arcsin', 'arccos', 'arctan'}

# 反三角函数的奇偶性
_ODD_INV_TRIG = {'arcsin', 'arctan'}
_EVEN_INV_TRIG = set()  # arccos 不是偶函数


def _is_pi(node):
    """判断是否为 π 常量"""
    return node.type == 'const_pi'


def _is_pi_over(node, n):
    """判断是否为 π/n (如 π/2, π/3)"""
    if node.type == 'div' and _is_pi(node.children[0]) and is_num(node.children[1], n):
        return True
    return False


def _is_n_times_pi(node, n):
    """判断是否为 n*π (如 2π, 3π)"""
    if node.type == 'mul' and is_num(node.children[0], n) and _is_pi(node.children[1]):
        return True
    if node.type == 'mul' and _is_pi(node.children[0]) and is_num(node.children[1], n):
        return True
    return False


def _match_pi_plus_x(arg):
    """
    匹配 π + x 或 x + π, 返回 (x, n) 或 None。
    n 为 π 的倍数 (1 或 2)。
    """
    if arg.type == 'add':
        a, b = arg.children[0], arg.children[1]
        if _is_pi(a):
            return b, 1
        if _is_pi(b):
            return a, 1
        if _is_n_times_pi(a, 2):
            return b, 2
        if _is_n_times_pi(b, 2):
            return a, 2
    return None


def _match_pi_minus_x(arg):
    """匹配 π - x, 返回 x 或 None"""
    if arg.type == 'sub' and _is_pi(arg.children[0]):
        return arg.children[1]
    return None


def _match_pi_over2_plus_x(arg):
    """匹配 π/2 + x 或 x + π/2, 返回 x 或 None"""
    if arg.type == 'add':
        a, b = arg.children[0], arg.children[1]
        if _is_pi_over(a, 2):
            return b
        if _is_pi_over(b, 2):
            return a
    return None


def _match_pi_over2_minus_x(arg):
    """匹配 π/2 - x, 返回 x 或 None"""
    if arg.type == 'sub' and _is_pi_over(arg.children[0], 2):
        return arg.children[1]
    return None


def _match_2pi_minus_x(arg):
    """匹配 2π - x, 返回 x 或 None"""
    if arg.type == 'sub' and _is_n_times_pi(arg.children[0], 2):
        return arg.children[1]
    return None


def _simplify_trig(node):
    """
    三角函数化简:
      1. 奇偶性: sin(-x)=-sin(x), cos(-x)=cos(x), ...
      2. 诱导公式: sin(π-x)=sin(x), cos(π/2-x)=sin(x), ...
      3. 周期性: sin(x+2π)=sin(x), tan(x+π)=tan(x), ...
    """
    t = node.type
    arg = node.children[0]

    # ── 1. 奇偶性 ──
    if arg.type == 'neg':
        inner = arg.children[0]
        if t in _ODD_TRIG:
            return Neg(SNode(t, inner))
        if t in _EVEN_TRIG:
            return SNode(t, inner)
        if t in _ODD_INV_TRIG:
            return Neg(SNode(t, inner))

    # ── 2. 诱导公式 (仅 sin/cos) ──
    if t in ('sin', 'cos'):
        # sin(π - x) = sin(x),  cos(π - x) = -cos(x)
        x = _match_pi_minus_x(arg)
        if x is not None:
            if t == 'sin':
                return Sin(x)
            else:
                return Neg(Cos(x))

        # sin(π + x) = -sin(x),  cos(π + x) = -cos(x)
        # sin(x + 2π) = sin(x),  cos(x + 2π) = cos(x) (周期性)
        m = _match_pi_plus_x(arg)
        if m is not None:
            x, n = m
            if n == 1:
                if t == 'sin':
                    return Neg(Sin(x))
                else:
                    return Neg(Cos(x))
            elif n == 2:
                return SNode(t, x)

        # sin(π/2 - x) = cos(x),  cos(π/2 - x) = sin(x)
        x = _match_pi_over2_minus_x(arg)
        if x is not None:
            if t == 'sin':
                return Cos(x)
            else:
                return Sin(x)

        # sin(π/2 + x) = cos(x),  cos(π/2 + x) = -sin(x)
        x = _match_pi_over2_plus_x(arg)
        if x is not None:
            if t == 'sin':
                return Cos(x)
            else:
                return Neg(Sin(x))

        # sin(2π - x) = -sin(x),  cos(2π - x) = cos(x)
        x = _match_2pi_minus_x(arg)
        if x is not None:
            if t == 'sin':
                return Neg(Sin(x))
            else:
                return Cos(x)

    # ── 3. 周期性 (tan/cot 周期为 π) ──
    if t in ('tan', 'cot'):
        m = _match_pi_plus_x(arg)
        if m is not None:
            x, n = m
            if n == 1:
                return SNode(t, x)

    return node


def _simplify_trig_identities_add(node):
    """
    在加法中应用三角恒等式:
      sin²(x) + cos²(x) = 1
      1 + tan²(x) = sec²(x)
      1 + cot²(x) = csc²(x)
    返回化简后的节点, 或 None。
    """
    if node.type != 'add':
        return None
    a, b = node.children[0], node.children[1]

    def _is_trig_sq(n, trig_type):
        return (n.type == 'pow' and n.children[0].type == trig_type
                and is_num(n.children[1], 2))

    # sin²(x) + cos²(x) = 1
    if _is_trig_sq(a, 'sin') and _is_trig_sq(b, 'cos'):
        if _structural_equal(a.children[0].children[0], b.children[0].children[0]):
            return ONE
    if _is_trig_sq(a, 'cos') and _is_trig_sq(b, 'sin'):
        if _structural_equal(a.children[0].children[0], b.children[0].children[0]):
            return ONE

    # 1 + tan²(x) = sec²(x)
    if is_num(a, 1) and _is_trig_sq(b, 'tan'):
        return Pow(SNode('sec', b.children[0].children[0]), TWO)
    if _is_trig_sq(a, 'tan') and is_num(b, 1):
        return Pow(SNode('sec', a.children[0].children[0]), TWO)

    # 1 + cot²(x) = csc²(x)
    if is_num(a, 1) and _is_trig_sq(b, 'cot'):
        return Pow(SNode('csc', b.children[0].children[0]), TWO)
    if _is_trig_sq(a, 'cot') and is_num(b, 1):
        return Pow(SNode('csc', a.children[0].children[0]), TWO)

    return None


def _simplify_trig_identities_mul(node):
    """
    在乘法中应用三角恒等式:
      tan(x) * cot(x) = 1
      sec(x) * cos(x) = 1
      csc(x) * sin(x) = 1
    """
    if node.type != 'mul':
        return None
    a, b = node.children[0], node.children[1]

    pairs = [('tan', 'cot'), ('cot', 'tan'),
             ('sec', 'cos'), ('cos', 'sec'),
             ('csc', 'sin'), ('sin', 'csc')]
    for t1, t2 in pairs:
        if a.type == t1 and b.type == t2:
            if _structural_equal(a.children[0], b.children[0]):
                return ONE
    return None


# =====================================================================
# 6. 高级函数识别 (三角函数/双曲函数/常数)
# =====================================================================

def recognize_functions(node: SNode) -> SNode:
    """在化简后的表达式中识别高级函数模式, 迭代到不动点"""
    prev = None
    curr = node
    for _ in range(30):
        curr = _recognize_once(curr)
        if prev is not None and _structural_equal(prev, curr):
            break
        prev = curr
    return curr


def _is_imag_unit(node):
    """判断是否为虚数单位 i 或 -i (仓库中 I() = -i)"""
    if node.type == 'const_i':
        return 1  # +i
    if node.type == 'neg' and node.children[0].type == 'const_i':
        return -1  # -i
    return 0


def _extract_imag_factor(expr):
    """
    从表达式中提取虚数单位因子, 支持各种嵌套形式:
      i*x, x*i, (i*x)/2, i*(x/2), 2*i*x, 等等
    返回 (imag_sign, rest) 或 None。rest 是去除虚数单位后的剩余表达式。
    """
    # 直接 mul(imag, rest) 或 mul(rest, imag)
    if expr.type == 'mul':
        for i, child in enumerate(expr.children):
            s = _is_imag_unit(child)
            if s != 0:
                rest = expr.children[1 - i]
                return (s, rest)
        # mul 中没有直接的 imag_unit, 但可能嵌套在子节点中 (如 mul(2, div(i*x, 3)))
        for i, child in enumerate(expr.children):
            result = _extract_imag_factor(child)
            if result is not None:
                s, rest_inner = result
                # 重建剩余部分: 用 rest_inner 替换原来的 child
                other = expr.children[1 - i]
                new_rest = Mul(rest_inner, other)
                return (s, new_rest)
    # div(mul(imag, rest), n) = (imag*rest)/n
    if expr.type == 'div':
        result = _extract_imag_factor(expr.children[0])
        if result is not None:
            s, rest_inner = result
            new_rest = Div(rest_inner, expr.children[1])
            return (s, new_rest)
    # neg(expr) — 负号不影响虚数单位提取
    if expr.type == 'neg':
        result = _extract_imag_factor(expr.children[0])
        if result is not None:
            s, rest = result
            return (s, Neg(rest))
    return None


def _is_negation(a, b):
    """
    判断 b 是否为 a 的相反数, 支持各种形式:
      neg(a), mul(-c, rest) vs mul(c, rest), a vs neg(a), 等等
    """
    # 直接: b = neg(a)
    if b.type == 'neg' and _structural_equal(a, b.children[0]):
        return True
    if a.type == 'neg' and _structural_equal(b, a.children[0]):
        return True
    # mul(-c, rest) vs mul(c, rest)
    if a.type == 'mul' and b.type == 'mul':
        a_num = None
        a_rest = None
        b_num = None
        b_rest = None
        for child in a.children:
            if child.type == 'num':
                a_num = child.value
            else:
                a_rest = child
        for child in b.children:
            if child.type == 'num':
                b_num = child.value
            else:
                b_rest = child
        if a_num is not None and b_num is not None and a_num == -b_num:
            if (a_rest is None and b_rest is None) or \
               (a_rest is not None and b_rest is not None and _structural_equal(a_rest, b_rest)):
                return True
    # a vs mul(-1, a)
    if b.type == 'mul' and len(b.children) == 2:
        if is_num(b.children[0], -1) and _structural_equal(a, b.children[1]):
            return True
        if is_num(b.children[1], -1) and _structural_equal(a, b.children[0]):
            return True
    if a.type == 'mul' and len(a.children) == 2:
        if is_num(a.children[0], -1) and _structural_equal(b, a.children[1]):
            return True
        if is_num(a.children[1], -1) and _structural_equal(b, a.children[0]):
            return True
    return False


def _extract_exp_pair(node):
    """
    从加法/减法节点中提取 exp(A) 和 exp(B) 对, 其中 B = -A (或 A = -B)。
    返回 (arg, sign) 其中:
      arg 是指数中的非虚数部分 (如 x)
      sign = +1 表示 exp(A) + exp(-A) 或 exp(A) - exp(-A) 正向
      sign = -1 表示 exp(-A) - exp(A) (反向)
    imag_flag: 1 表示含虚数单位 (三角函数), 0 表示不含 (双曲函数)
    """
    if node.type not in ('add', 'sub'):
        return None
    ex1, ex2 = node.children[0], node.children[1]
    if ex1.type != 'exp' or ex2.type != 'exp':
        return None
    a1, a2 = ex1.children[0], ex2.children[0]

    # 确定正指数和负指数 (正指数是不含外层负号的那个)
    pos_exp = None
    neg_exp = None
    if _is_negation(a1, a2):
        if a1.type == 'neg' and a2.type != 'neg':
            pos_exp, neg_exp = a2, a1
        else:
            pos_exp, neg_exp = a1, a2
    elif _is_negation(a2, a1):
        if a2.type == 'neg' and a1.type != 'neg':
            pos_exp, neg_exp = a1, a2
        else:
            pos_exp, neg_exp = a2, a1
    else:
        return None

    # 如果正指数的虚数单位为 -i (如 mul(-i, x)), 交换正负指数
    # 这样 pos_exp 总是 i*arg 形式, 保证符号正确
    imag_result = _extract_imag_factor(pos_exp)
    if imag_result is not None:
        imag_sign, _ = imag_result
        if imag_sign == -1:
            pos_exp, neg_exp = neg_exp, pos_exp

    # 分析正指数: 可能是 i*x, (i*x)/2, 2*i*x 等各种形式
    imag_result = _extract_imag_factor(pos_exp)
    if imag_result is not None:
        imag_sign, arg = imag_result
        is_trig = True
    else:
        # 双曲函数: pos_exp = arg (不含虚数单位)
        arg = pos_exp
        is_trig = False
        imag_sign = 0

    # 确定符号
    if node.type == 'add':
        sign = 1  # exp(A) + exp(-A)
    else:  # sub
        if ex1.type == 'exp' and _structural_equal(ex1.children[0], pos_exp):
            sign = 1  # exp(A) - exp(-A)
        else:
            sign = -1  # exp(-A) - exp(A)

    return (arg, sign, is_trig)


def _is_imag_unit_factor(node):
    """如果 node 是 imag_unit * something 的形式, 返回 imag_unit 的符号 (+1 或 -1), 否则返回 0"""
    if node.type != 'mul':
        return 0
    for child in node.children:
        s = _is_imag_unit(child)
        if s != 0:
            return s
    return 0


def _get_other_factor(mul_node, imag_sign):
    """从 mul(imag_unit, arg) 中提取 arg"""
    for child in mul_node.children:
        s = _is_imag_unit(child)
        if s == 0:
            return child
    return None


def _try_recognize_trig(node):
    """
    尝试从各种等价形式中识别 sin/cos/sinh/cosh。
    处理的形式包括:
      (exp(A) ± exp(-A)) / 2
      (exp(A) ± exp(-A)) / (2*imag_unit)
      -(1/(2*imag_unit) * (exp(-A) - exp(A)))
      (exp(A) ± exp(-A)) * 0.5
      等等
    """
    t = node.type

    # 形式1: div(numerator, denominator)
    if t == 'div':
        numerator = node.children[0]
        denominator = node.children[1]
        # 正函数: (exp_pair) / denom
        result = _match_trig_div(numerator, denominator)
        if result is not None:
            return result
        # 倒数函数: 2 / (exp_pair) = sec / sech / csch
        if is_num(numerator, 2) and denominator.type in ('add', 'sub'):
            pair_info = _extract_exp_pair(denominator)
            if pair_info is not None:
                arg, sign, is_trig = pair_info
                if is_trig and denominator.type == 'add':
                    return SNode('sec', arg)
                if not is_trig and denominator.type == 'add':
                    return SNode('sech', arg)
                if not is_trig and denominator.type == 'sub':
                    # csch = 2/(exp(x)-exp(-x)); sign 处理
                    if sign == 1:
                        return SNode('csch', arg)
                    else:
                        return Neg(SNode('csch', arg))
        # 倒数函数: (2*imag_unit) / (exp_pair) = csc
        if denominator.type == 'sub':
            factor = _extract_2i_factor(numerator)
            if factor is not None:
                num_sign, imag_sign = factor
                pair_info = _extract_exp_pair(denominator)
                if pair_info is not None:
                    arg, sign, is_trig = pair_info
                    if is_trig:
                        total = num_sign * imag_sign * sign
                        if total == 1:
                            return SNode('csc', arg)
                        else:
                            return Neg(SNode('csc', arg))

    # 形式2: neg(mul(factor, exp_pair))  — 如 -(1/(2i) * (exp(-ix) - exp(ix)))
    if t == 'neg':
        inner = node.children[0]
        if inner.type == 'mul':
            # 找到 exp_pair 因子和分母因子
            exp_pair = None
            denom_factor = None
            for child in inner.children:
                if child.type in ('add', 'sub') and child.children[0].type == 'exp':
                    exp_pair = child
                else:
                    denom_factor = child
            if exp_pair is not None and denom_factor is not None:
                # 整体是 -(denom_factor * exp_pair) = (-denom_factor) * exp_pair
                # 等价于 div(exp_pair, 1/(-denom_factor))
                # 直接尝试匹配: 分母 = 1/(-denom_factor) = -1/denom_factor
                neg_denom = Neg(denom_factor)
                result = _match_trig_div(exp_pair, Div(ONE, neg_denom))
                if result is not None:
                    return result
                # 也尝试: 分母 = 1/denom_factor (不带负号)
                result = _match_trig_div(exp_pair, Div(ONE, denom_factor))
                if result is not None:
                    # 但整体有负号, 所以结果取反
                    if result.type == 'sin':
                        return Neg(result)
                    return result

    # 形式3: mul(numerator, factor) where factor = 1/denominator
    if t == 'mul':
        exp_pair = None
        recip_factor = None
        for child in node.children:
            if child.type in ('add', 'sub') and len(child.children) >= 2 and child.children[0].type == 'exp':
                exp_pair = child
            elif child.type == 'div' and is_num(child.children[0], 1):
                recip_factor = child
        if exp_pair is not None and recip_factor is not None:
            denominator = recip_factor.children[1]
            return _match_trig_div(exp_pair, denominator)

    return None


def _match_trig_div(numerator, denominator):
    """匹配 (exp_pair) / denominator 形式的三角函数"""
    pair_info = _extract_exp_pair(numerator)
    if pair_info is None:
        return None
    arg, sign, is_trig = pair_info

    if not is_trig:
        # 双曲函数: 分母应为 2
        if is_num(denominator, 2):
            if sign == 1 and numerator.type == 'add':
                return Cosh(arg)
            if sign == 1 and numerator.type == 'sub':
                return Sinh(arg)
            if sign == -1:  # exp(-A) - exp(A) = -(exp(A) - exp(-A))
                return Neg(Sinh(arg))
        return None

    # 三角函数
    # cos(x) = (exp(i*x) + exp(-i*x)) / 2  — 分母为 2
    if numerator.type == 'add' and is_num(denominator, 2):
        return Cos(arg)

    # sin(x) = (exp(i*x) - exp(-i*x)) / (2 * imag_unit)
    if numerator.type == 'sub':
        factor = _extract_2i_factor(denominator)
        if factor is None:
            return None
        num_sign, imag_sign = factor
        # sign=1: exp(A) - exp(-A); sign=-1: exp(-A) - exp(A)
        # 总符号: sign * num_sign * imag_sign
        total_sign = sign * num_sign * imag_sign
        if total_sign == 1:
            return Sin(arg)
        else:
            return Neg(Sin(arg))

    return None


def _denominator_has_2i(denominator):
    """
    检查分母是否为 2 * imag_unit 的形式。
    返回 imag_unit 的符号 (+1 或 -1), 不是则返回 0。
    """
    if denominator.type != 'mul':
        return 0
    has_2 = False
    imag_sign = 0
    for child in denominator.children:
        if is_num(child, 2):
            has_2 = True
        s = _is_imag_unit(child)
        if s != 0:
            imag_sign = s
    if has_2 and imag_sign != 0:
        return imag_sign
    return 0


def _extract_2i_factor(node):
    """
    从节点中提取 2*imag_unit 因子, 支持各种等价形式:
      2*i, -2*i, 2*(-i), i*2, (-2)*i, -(2*i), 等等
    返回 (num_sign, imag_sign) 或 None。
    整体值 = num_sign * 2 * imag_sign * i  (即 num_sign*imag_sign * 2i)
    """
    inner = node
    num_sign = 1
    if node.type == 'neg':
        inner = node.children[0]
        num_sign = -1
    if inner.type != 'mul':
        return None
    num_val = None
    imag_sign = 0
    for child in inner.children:
        if child.type == 'num':
            num_val = child.value
        s = _is_imag_unit(child)
        if s != 0:
            imag_sign = s
    if num_val is None or imag_sign == 0:
        return None
    # num_val 应该是 ±2
    if abs(num_val) != 2:
        return None
    if num_val < 0:
        num_sign *= -1
    return (num_sign, imag_sign)
    """
    检查分母是否为 2 * imag_unit 的形式。
    返回 imag_unit 的符号 (+1 或 -1), 不是则返回 0。
    """
    if denominator.type != 'mul':
        return 0
    has_2 = False
    imag_sign = 0
    for child in denominator.children:
        if is_num(child, 2):
            has_2 = True
        s = _is_imag_unit(child)
        if s != 0:
            imag_sign = s
    if has_2 and imag_sign != 0:
        return imag_sign
    return 0


def _match_arcsin(expr):
    """匹配 sqrt(1-x^2) - i*x, 返回 x 或 None"""
    if expr.type != 'sub':
        return None
    a, b = expr.children[0], expr.children[1]
    # a = sqrt(1-x^2), b = i*x
    if a.type == 'sqrt':
        inner = a.children[0]
        if inner.type == 'sub' and is_num(inner.children[0], 1) and inner.children[1].type == 'pow':
            base = inner.children[1].children[0]
            if is_num(inner.children[1].children[1], 2):
                # b should be i * base (or base * i)
                if b.type == 'mul':
                    for child in b.children:
                        if _is_imag_unit(child) != 0:
                            other = b.children[1 - b.children.index(child)]
                            if _structural_equal(other, base):
                                return base
    return None


def _match_arccos(expr):
    """匹配 x - i*sqrt(1-x^2), 返回 x 或 None"""
    if expr.type != 'sub':
        return None
    a, b = expr.children[0], expr.children[1]
    # a = x, b = i*sqrt(1-x^2)
    if b.type == 'mul':
        imag_child = None
        other_child = None
        for child in b.children:
            if _is_imag_unit(child) != 0:
                imag_child = child
            else:
                other_child = child
        if imag_child is not None and other_child is not None and other_child.type == 'sqrt':
            inner = other_child.children[0]
            if inner.type == 'sub' and is_num(inner.children[0], 1) and inner.children[1].type == 'pow':
                base = inner.children[1].children[0]
                if is_num(inner.children[1].children[1], 2) and _structural_equal(a, base):
                    return base
    return None


def _match_arctan(expr):
    """匹配 (i-x)/(i+x) 或等价形式 (-i-x)/(x-i), 返回 x 或 None"""
    if expr.type != 'div':
        return None
    num, den = expr.children[0], expr.children[1]
    # num = i - x (or -i - x), den = i + x (or x - i)
    # Try: num = sub(i, x), den = add(i, x)
    def _extract_ix_sub(node):
        """从 i-x 或 -i-x 中提取 x, 返回 (x, imag_sign)"""
        if node.type == 'sub':
            a, b = node.children[0], node.children[1]
            s = _is_imag_unit(a)
            if s != 0:
                return (b, s)
        if node.type == 'add':
            # -i + (-x) = -i - x
            a, b = node.children[0], node.children[1]
            if _is_imag_unit(a) != 0 and b.type == 'neg':
                return (b.children[0], _is_imag_unit(a))
            if _is_imag_unit(b) != 0 and a.type == 'neg':
                return (a.children[0], _is_imag_unit(b))
        return None
    def _extract_ix_add(node):
        """从 i+x 或 x-i 中提取 x, 返回 (x, imag_sign)"""
        if node.type == 'add':
            a, b = node.children[0], node.children[1]
            s = _is_imag_unit(a)
            if s != 0:
                return (b, s)
            s = _is_imag_unit(b)
            if s != 0:
                return (a, s)
        if node.type == 'sub':
            # x - i = x + (-i)
            a, b = node.children[0], node.children[1]
            s = _is_imag_unit(b)
            if s != 0:
                return (a, -s)  # x - i = x + (-i), imag_sign = -s
        return None
    num_info = _extract_ix_sub(num)
    den_info = _extract_ix_add(den)
    if num_info is not None and den_info is not None:
        x1, s1 = num_info
        x2, s2 = den_info
        if _structural_equal(x1, x2) and s1 == s2:
            return x1
    return None


def _recognize_once(node):
    # 先递归子节点
    if node.type in ('var', 'num', 'const_e', 'const_pi', 'const_i'):
        return node
    children = [_recognize_once(c) for c in node.children]
    node = SNode(node.type, *children, value=node.value)

    t = node.type

    # 常数 e = exp(1) 已在 _simplify_exp 中处理
    # 常数识别
    # pi = i * ln(-1)  (数学上 ln(-1) = i*pi, 所以 i * ln(-1) = -pi)
    # 仓库中 PI() = I() * LN(-1), 而 I() = -i, 所以 (-i) * ln(-1) = pi
    if t == 'mul':
        a, b = node.children[0], node.children[1]
        # (-i) * ln(-1) = pi
        if a.type == 'neg' and a.children[0].type == 'const_i' and b.type == 'ln' and b.children[0] == NEG_ONE:
            return PI_CONST
        if b.type == 'neg' and b.children[0].type == 'const_i' and a.type == 'ln' and a.children[0] == NEG_ONE:
            return PI_CONST
        # i * ln(-1) = -pi
        if a.type == 'const_i' and b.type == 'ln' and b.children[0] == NEG_ONE:
            return Neg(PI_CONST)
        if b.type == 'const_i' and a.type == 'ln' and a.children[0] == NEG_ONE:
            return Neg(PI_CONST)

    # ---- 三角函数识别 (灵活匹配各种等价形式) ----
    result = _try_recognize_trig(node)
    if result is not None:
        return result

    # tan(x) = sin(x)/cos(x) = sin(x)*sec(x), tanh(x) = sinh(x)/cosh(x) = sinh(x)*sech(x)
    if t == 'div':
        a, b = node.children[0], node.children[1]
        if a.type == 'sin' and b.type == 'cos' and _structural_equal(a.children[0], b.children[0]):
            return Tan(a.children[0])
        if a.type == 'sinh' and b.type == 'cosh' and _structural_equal(a.children[0], b.children[0]):
            return Tanh(a.children[0])
    if t == 'mul':
        a, b = node.children[0], node.children[1]
        # sin * sec = tan
        if a.type == 'sin' and b.type == 'sec' and _structural_equal(a.children[0], b.children[0]):
            return Tan(a.children[0])
        if b.type == 'sin' and a.type == 'sec' and _structural_equal(a.children[0], b.children[0]):
            return Tan(a.children[0])
        # cos * csc = cot
        if a.type == 'cos' and b.type == 'csc' and _structural_equal(a.children[0], b.children[0]):
            return SNode('cot', a.children[0])
        if b.type == 'cos' and a.type == 'csc' and _structural_equal(a.children[0], b.children[0]):
            return SNode('cot', a.children[0])
        # sinh * sech = tanh
        if a.type == 'sinh' and b.type == 'sech' and _structural_equal(a.children[0], b.children[0]):
            return Tanh(a.children[0])
        if b.type == 'sinh' and a.type == 'sech' and _structural_equal(a.children[0], b.children[0]):
            return Tanh(a.children[0])
        # cosh * csch = coth
        if a.type == 'cosh' and b.type == 'csch' and _structural_equal(a.children[0], b.children[0]):
            return SNode('coth', a.children[0])
        if b.type == 'cosh' and a.type == 'csch' and _structural_equal(a.children[0], b.children[0]):
            return SNode('coth', a.children[0])

    # sqrt(x^2) = |x|
    if t == 'sqrt' and node.children[0].type == 'pow' and is_num(node.children[0].children[1], 2):
        return Abs(node.children[0].children[0])

    # ---- 反三角函数识别 ----
    # arcsin(x) = i * ln(sqrt(1-x^2) - i*x)
    if t == 'mul':
        a, b = node.children[0], node.children[1]
        # i * ln(...)
        if _is_imag_unit(a) != 0 and b.type == 'ln':
            result = _match_arcsin(b.children[0])
            if result is not None:
                return SNode('arcsin', result)
            result = _match_arccos(b.children[0])
            if result is not None:
                return SNode('arccos', result)
        if _is_imag_unit(b) != 0 and a.type == 'ln':
            result = _match_arcsin(a.children[0])
            if result is not None:
                return SNode('arcsin', result)
            result = _match_arccos(a.children[0])
            if result is not None:
                return SNode('arccos', result)

    # arctan(x) = -(i/2 * ln((i-x)/(i+x)))  [with I()=-i: -(i/2 * ln((-i-x)/(x-i)))]
    if t == 'neg':
        inner = node.children[0]
        if inner.type == 'mul':
            # find i/2 factor and ln factor
            half_i_factor = None
            ln_factor = None
            for child in inner.children:
                if child.type == 'ln':
                    ln_factor = child
                elif child.type == 'div' and _is_imag_unit(child.children[0]) != 0 and is_num(child.children[1], 2):
                    half_i_factor = child
                elif child.type == 'mul' and len(child.children) == 2:
                    # could be i/2 represented as i*(1/2) or (1/2)*i
                    c1, c2 = child.children
                    if (_is_imag_unit(c1) != 0 and _is_half(c2)) or (_is_imag_unit(c2) != 0 and _is_half(c1)):
                        half_i_factor = child
            if half_i_factor is not None and ln_factor is not None:
                result = _match_arctan(ln_factor.children[0])
                if result is not None:
                    return SNode('arctan', result)

    return node


def _match_cos_pair(ex1, ex2):
    """匹配 exp(i*x) + exp(-i*x) 返回 x, 否则 None"""
    if ex1.type != 'exp' or ex2.type != 'exp':
        return None
    a1, a2 = ex1.children[0], ex2.children[0]
    # a1 = i*x, a2 = -i*x = neg(i*x)
    if a2.type == 'neg' and _structural_equal(a1, a2.children[0]):
        # a1 应该是 i * x
        if a1.type == 'mul' and a1.children[0].type == 'const_i':
            return a1.children[1]
        if a1.type == 'mul' and a1.children[1].type == 'const_i':
            return a1.children[0]
    # 对称
    if a1.type == 'neg' and _structural_equal(a2, a1.children[0]):
        if a2.type == 'mul' and a2.children[0].type == 'const_i':
            return a2.children[1]
        if a2.type == 'mul' and a2.children[1].type == 'const_i':
            return a2.children[0]
    return None


def _match_sin_pair(ex1, ex2):
    """匹配 exp(i*x) - exp(-i*x) 返回 x"""
    return _match_cos_pair(ex1, ex2)  # 结构相同


def _match_cosh_pair(ex1, ex2):
    """匹配 exp(x) + exp(-x) 返回 x"""
    if ex1.type != 'exp' or ex2.type != 'exp':
        return None
    a1, a2 = ex1.children[0], ex2.children[0]
    if a2.type == 'neg' and _structural_equal(a1, a2.children[0]):
        return a1
    if a1.type == 'neg' and _structural_equal(a2, a1.children[0]):
        return a2
    return None


def _match_sinh_pair(ex1, ex2):
    """匹配 exp(x) - exp(-x) 返回 x"""
    return _match_cosh_pair(ex1, ex2)


# =====================================================================
# 7. 美化输出 (标准数学记号)
# =====================================================================

# 运算符优先级
_PREC = {
    'add': 1, 'sub': 1,
    'mul': 2, 'div': 2,
    'neg': 3,
    'pow': 4,
    'exp': 5, 'ln': 5,
    'sin': 5, 'cos': 5, 'tan': 5, 'cot': 5, 'sec': 5, 'csc': 5,
    'sinh': 5, 'cosh': 5, 'tanh': 5, 'coth': 5, 'sech': 5, 'csch': 5,
    'arcsin': 5, 'arccos': 5, 'arctan': 5,
    'sqrt': 5, 'abs': 5,
    'eml': 0,
}


def to_math(node: SNode, parent_prec: int = 0) -> str:
    """将 SNode 转换为标准数学记号字符串"""
    t = node.type

    if t == 'num':
        v = node.value
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    if t == 'var':
        return 'x'

    if t == 'const_e':
        return 'e'
    if t == 'const_pi':
        return 'pi'
    if t == 'const_i':
        return 'i'

    if t == 'neg':
        inner = to_math(node.children[0], _PREC['neg'])
        s = f"-{inner}"
        return _wrap(s, parent_prec, _PREC['neg'])

    if t == 'add':
        a = to_math(node.children[0], _PREC['add'])
        b = to_math(node.children[1], _PREC['add'] + 1)  # +1 避免 a + (b + c) 多余括号
        s = f"{a} + {b}"
        return _wrap(s, parent_prec, _PREC['add'])

    if t == 'sub':
        a = to_math(node.children[0], _PREC['sub'])
        b = to_math(node.children[1], _PREC['sub'] + 1)
        s = f"{a} - {b}"
        return _wrap(s, parent_prec, _PREC['sub'])

    if t == 'mul':
        a = to_math(node.children[0], _PREC['mul'])
        b = to_math(node.children[1], _PREC['mul'] + 1)
        s = f"{a}*{b}"
        return _wrap(s, parent_prec, _PREC['mul'])

    if t == 'div':
        a = to_math(node.children[0], _PREC['div'])
        b = to_math(node.children[1], _PREC['div'] + 1)
        s = f"{a}/{b}"
        return _wrap(s, parent_prec, _PREC['div'])

    if t == 'pow':
        base = node.children[0]
        # 负数或低优先级底数需要括号
        a = to_math(base, _PREC['pow'] + 1)
        if base.type == 'neg' or (base.type == 'num' and base.value < 0):
            a = f"({a})"
        b = to_math(node.children[1], _PREC['pow'])
        s = f"{a}^{b}"
        return _wrap(s, parent_prec, _PREC['pow'])

    # 函数调用
    func_names = {
        'exp': 'exp', 'ln': 'ln',
        'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
        'cot': 'cot', 'sec': 'sec', 'csc': 'csc',
        'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh',
        'coth': 'coth', 'sech': 'sech', 'csch': 'csch',
        'sqrt': 'sqrt', 'abs': 'abs',
        'arcsin': 'arcsin', 'arccos': 'arccos', 'arctan': 'arctan',
    }
    if t in func_names:
        arg = to_math(node.children[0], 0)
        return f"{func_names[t]}({arg})"

    if t == 'eml':
        a = to_math(node.children[0], 0)
        b = to_math(node.children[1], 0)
        return f"f({a}, {b})"

    return repr(node)


def _wrap(s, parent_prec, my_prec):
    """如果父运算符优先级更高, 则加括号"""
    if parent_prec > my_prec:
        return f"({s})"
    return s


# =====================================================================
# 8. 主入口
# =====================================================================

def reverse_eml(expr, from_repo_node=False) -> str:
    """
    从 EML 表达式逆向化简回原始数学表达式。

    Args:
        expr: EML 表达式字符串 (如 "f(X, 1)"), 或仓库的 Node 对象 (from_repo_node=True)
        from_repo_node: 若为 True, expr 被视为仓库 Node 对象

    Returns:
        化简后的标准数学表达式字符串
    """
    if from_repo_node:
        raw = node_to_snode(expr)
    else:
        raw = parse_eml(expr)

    # Step 1: 模式提升
    lifted = lift(raw)

    # Step 2: 交替执行代数化简和函数识别 (避免化简破坏识别结构)
    curr = lifted
    for _ in range(15):
        prev = curr
        curr = simplify(curr)
        curr = recognize_functions(curr)
        if _structural_equal(prev, curr):
            break

    # Step 3: 最终化简
    final = simplify(curr)

    return to_math(final)


def reverse_from_original(expr_str: str) -> str:
    """
    便捷函数: 先正向编译原始表达式为 EML, 再逆向化简回来。
    用于测试逆向化简器的正确性。
    """
    import sys
    sys.path.insert(0, '/home/user/.super_doubao/super-doubao-runtime/workspace/eml-converter')
    from eml import parse_expr as repo_parse
    node = repo_parse(expr_str)
    return reverse_eml(node, from_repo_node=True)


# =====================================================================
# 9. 自测
# =====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("EML 逆向化简器测试")
    print("=" * 60)

    # 基础 EML 模式测试
    tests = [
        ("f(X, 1)", "exp(x)"),
        ("f(1, f(f(1, X), 1))", "ln(x)"),
        ("f(1, 1)", "e"),
        ("f(f(1, f(f(1, X), 1)), f(1, 1))", "x - 1"),  # SUB(x,1)
    ]

    print("\n--- 直接 EML 表达式测试 ---")
    for eml_str, expected in tests:
        result = reverse_eml(eml_str)
        status = "OK" if result == expected else "??"
        print(f"[{status}] {eml_str[:60]:60s} => {result}  (expected: {expected})")

    # 从原始表达式往返测试
    print("\n--- 原始表达式 → EML → 逆向化简 往返测试 ---")
    roundtrip = [
        "X",
        "1",
        "EXP(X)",
        "LN(X)",
        "X+1",
        "X-1",
        "1-X",
        "2*X",
        "X/2",
        "X^2",
        "X^3",
        "3*X+2",
        "X*X",
        "SIN(X)",
        "COS(X)",
        "EXP(X)+1",
        "LN(X)+X",
        "X+X",
        "2",
        "3",
        "X-(-1)",
    ]

    for orig in roundtrip:
        try:
            result = reverse_from_original(orig)
            print(f"  {orig:15s} => {result}")
        except Exception as e:
            print(f"  {orig:15s} => ERROR: {e}")
