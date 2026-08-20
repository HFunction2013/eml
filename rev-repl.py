#!/usr/bin/env python3
"""
EML 逆向化简器 REPL  v2.0

功能:
  - 输入数学表达式 → 返回化简后的标准数学表达式 + EML 解析树
  - 输入 EML 表达式 (f(a,b) 格式) → 逆向化简
  - with x=表达式  子句 (兼容原仓库)
  - source <文件>   从文件读取并执行指令
  - let x=值        设定变量 x 的值 (用于求值)
  - eval 表达式     在当前 x 值下求值
  - \c              清空当前行 (兼容原仓库)
  - test            运行内置测试
  - help            帮助

用法:
  python3 repl.py                  # 交互模式
  python3 repl.py --eml            # EML 输入模式
  python3 repl.py "SIN(X)+1"      # 单次执行
  python3 repl.py --source script.txt  # 执行脚本文件
"""

import sys
import os
import cmath

# 确保能导入 reverse_eml 和原仓库
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "eml-converter"))

from reverse_eml import reverse_eml, reverse_from_original, parse_eml, node_to_snode, lift, simplify, recognize_functions, to_math

# 原仓库模块 (用于 with 子句求值和 parse_expr)
try:
    from eml import evaluate_expr, parse_expr as repo_parse_expr
    _HAS_REPO = True
except ImportError:
    _HAS_REPO = False


# ═══════════════════════════════════════════════════════════
#  常量与样式
# ═══════════════════════════════════════════════════════════

BANNER = r"""
╔════════════════════════════════════════════════════════╗
║          EML 逆向化简器 REPL  v2.0                     ║
║  表达式 → 化简结果 | source 执行文件 | with x= 求值    ║
║  输入 help 查看命令, exit 退出                          ║
╚════════════════════════════════════════════════════════╝
"""

# 颜色 (简单 ANSI, 不依赖第三方库)
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[31m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    BLUE   = "\033[34m"
    CYAN   = "\033[36m"
    GREY   = "\033[90m"

# 自动检测是否支持颜色 (非 TTY 时禁用)
_USE_COLOR = sys.stdout.isatty()

def _c(text, color):
    if not _USE_COLOR:
        return text
    return f"{color}{text}{C.RESET}"


# ═══════════════════════════════════════════════════════════
#  内置测试
# ═══════════════════════════════════════════════════════════

TEST_CASES = [
    ("X", "x"), ("X+1", "x + 1"), ("X-1", "x - 1"),
    ("1-X", "1 - x"), ("2*X", "2*x"), ("X/2", "x/2"),
    ("X^2", "x^2"), ("X^3", "x^3"), ("3*X+2", "3*x + 2"),
    ("EXP(X)", "exp(x)"), ("LN(X)", "ln(x)"),
    ("SIN(X)", "sin(x)"), ("COS(X)", "cos(x)"),
    ("TAN(X)", "tan(x)"), ("COT(X)", "cot(x)"),
    ("SEC(X)", "sec(x)"), ("CSC(X)", "csc(x)"),
    ("SINH(X)", "sinh(x)"), ("COSH(X)", "cosh(x)"),
    ("TANH(X)", "tanh(x)"),
    ("ARCSIN(X)", "arcsin(x)"), ("ARCCOS(X)", "arccos(x)"),
    ("ARCTAN(X)", "arctan(x)"),
    ("SQRT(X)", "sqrt(x)"), ("ABS(X)", "abs(x)"),
    ("PI()", "pi"), ("E()", "e"), ("PHI()", "(1 + sqrt(5))/2"),
    ("X*X", "x^2"), ("X+X", "2*x"), ("X-X", "0"), ("X/X", "1"),
    ("SIN(2*X)", "sin(2*x)"), ("COS(X/2)", "cos(x/2)"),
    ("SIN(X+1)", "sin(x + 1)"),
    ("LOG2(X)", "ln(x)/ln(2)"), ("LOG10(X)", "ln(x)/ln(10)"),
    ("LOGISTIC(X)", "1/(1 + exp(-x))"),
    ("EXP(-X)", "exp(-x)"), ("LN(1/X)", "-ln(x)"),
]


# ═══════════════════════════════════════════════════════════
#  核心化简 / 求值
# ═══════════════════════════════════════════════════════════

def simplify_expression(expr: str, mode: str = "math") -> str:
    """化简表达式, 返回标准数学记号字符串"""
    expr = expr.strip()
    if not expr:
        return ""
    if mode == "eml":
        return reverse_eml(expr)
    return reverse_from_original(expr)


def get_eml_tree(expr: str, mode: str = "math") -> str:
    """获取 EML 解析树的字符串表示"""
    if mode == "eml":
        raw = parse_eml(expr)
    else:
        if not _HAS_REPO:
            return "(原仓库 eml 模块不可用)"
        node = repo_parse_expr(expr)
        raw = node_to_snode(node)
    # 只 lift, 不化简, 展示原始 EML 结构
    lifted = lift(raw)
    return to_math(lifted)


def evaluate_at(expr: str, x_value: complex, mode: str = "math") -> complex:
    """在给定 x 值下求值 (使用原仓库的 evaluate_expr)"""
    if not _HAS_REPO:
        raise RuntimeError("原仓库 eml 模块不可用, 无法求值")
    if mode == "eml":
        # EML 模式: 解析为 EML 树后求值 (需要构造原仓库 Node)
        # 简化处理: 把 EML 表达式当作数学表达式的字符串形式不支持,
        # 所以 EML 模式下求值需要先逆向化简再用 Python eval
        simplified = reverse_eml(expr)
        return _python_eval(simplified, x_value)
    _, result = evaluate_expr(expr, x_value)
    return result


def _python_eval(expr_str: str, x: complex) -> complex:
    """用 Python eval 求值化简后的表达式 (用于 EML 模式回退)"""
    import math
    safe_globals = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
        "exp": math.exp, "ln": math.log, "log": math.log,
        "sqrt": math.sqrt, "abs": abs, "pi": math.pi, "e": math.e,
        "x": x, "X": x,
    }
    # 替换 ^ 为 **
    py_expr = expr_str.replace("^", "**")
    return eval(py_expr, {"__builtins__": {}}, safe_globals)


def split_with_clause(input_str: str):
    """
    分割表达式和 with 子句 (兼容原仓库 main.py 的逻辑)
    支持: 表达式 with x=表达式
    返回 (expr_str, value_expr) 或 (expr_str, None)
    """
    s = input_str.strip()
    s_lower = s.lower()
    with_pos = s_lower.find(" with ")
    if with_pos == -1:
        return s, None

    expr_str = s[:with_pos].strip()
    rest = s[with_pos + 5:].strip()
    if not rest:
        return expr_str, None

    # 检查 x= 或 x = 格式
    if rest[0].lower() == 'x' and len(rest) > 1 and rest[1] == '=':
        eq_pos = 1
    elif rest.lower().startswith("x ") and "=" in rest[1:]:
        eq_pos = rest.find("=")
    else:
        return expr_str, None

    value_expr = rest[eq_pos + 1:].strip()
    if not value_expr:
        return expr_str, None
    return expr_str, value_expr


def format_complex(z: complex, tol: float = 1e-6) -> str:
    """格式化复数输出, 清理微小误差"""
    real = z.real
    imag = z.imag
    if abs(real) < tol:
        real = 0.0
    if abs(imag) < tol:
        imag = 0.0
    if imag == 0:
        if isinstance(real, float) and real.is_integer():
            return str(int(real))
        return str(real)
    if real == 0:
        if imag == 1:
            return "i"
        if imag == -1:
            return "-i"
        return f"{imag}i"
    sign = "+" if imag > 0 else "-"
    imag_abs = abs(imag)
    if imag_abs == 1:
        return f"{real} {sign} i"
    return f"{real} {sign} {imag_abs}i"


# ═══════════════════════════════════════════════════════════
#  REPL 状态
# ═══════════════════════════════════════════════════════════

class ReplState:
    def __init__(self):
        self.mode = "math"           # "math" 或 "eml"
        self.x_value = complex(0)    # 当前 x 值
        self.show_tree = False       # 是否显示 EML 解析树
        self.show_simplified = True  # 是否显示化简结果
        self.variables = {}          # 用户变量
        self.history = []            # 历史记录
        self.script_depth = 0        # source 嵌套深度


# ═══════════════════════════════════════════════════════════
#  命令处理
# ═══════════════════════════════════════════════════════════

def cmd_help(state, args):
    print(f"""
{_c('可用命令:', C.BOLD)}
  {_c('exit / quit / q', C.CYAN)}        退出 REPL
  {_c('help / ?', C.CYAN)}              显示此帮助
  {_c('eml / math', C.CYAN)}            切换输入模式 (EML 表达式 / 数学表达式)
  {_c('mode', C.CYAN)}                  显示当前输入模式
  {_c('source <文件>', C.CYAN)}         从文件读取并执行指令
  {_c('let x=<值>', C.CYAN)}            设定 x 的值 (用于求值)
  {_c('x=<值>', C.CYAN)}                同上 (简写)
  {_c('vars', C.CYAN)}                  显示所有变量
  {_c('eval <表达式>', C.CYAN)}         在当前 x 值下求值
  {_c('tree on/off', C.CYAN)}           切换显示 EML 解析树
  {_c('test', C.CYAN)}                  运行内置测试 (40 项)
  {_c('history', C.CYAN)}               显示历史记录
  {_c('clear / cls', C.CYAN)}           清屏

{_c('表达式语法:', C.BOLD)}
  数学表达式: SIN(X), 2*X+1, EXP(X)*LN(X), X^2+3*X+2 ...
  EML 表达式:  f(X, 1), f(1, f(f(1, X), 1)) ...
  with 子句:    SIN(X) with x=PI()/4   (先求 x 值, 再代入主表达式)
  行尾 \\c:      清空当前行 (兼容原仓库)
""")


def cmd_test(state, args):
    print(f"\n{_c('运行内置测试...', C.BOLD)}")
    print("─" * 60)
    passed = failed = 0
    for expr, expected in TEST_CASES:
        try:
            result = simplify_expression(expr, "math")
            if result == expected:
                passed += 1
                print(f"  {_c('[PASS]', C.GREEN)} {expr:20s} => {result}")
            else:
                failed += 1
                print(f"  {_c('[FAIL]', C.RED)} {expr:20s} => {result}  "
                      f"{_c(f'(期望: {expected})', C.GREY)}")
        except Exception as e:
            failed += 1
            print(f"  {_c('[ERR ]', C.RED)} {expr:20s} => {e}")
    print("─" * 60)
    status = _c("全部通过!", C.GREEN) if failed == 0 else _c(f"{failed} 项失败", C.RED)
    print(f"结果: {passed} 通过, {failed} 失败, 共 {len(TEST_CASES)} 项 — {status}\n")
    return failed == 0


def cmd_source(state, args):
    """从文件读取并执行指令"""
    if not args:
        print(_c("用法: source <文件路径>", C.YELLOW))
        return
    filepath = args.strip()
    if not os.path.isfile(filepath):
        print(_c(f"文件不存在: {filepath}", C.RED))
        return
    if state.script_depth >= 5:
        print(_c("source 嵌套过深 (最大 5 层), 已停止", C.RED))
        return

    state.script_depth += 1
    print(_c(f"┌─ source: {filepath}", C.CYAN))
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # 跳过空行和注释
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                continue
            print(_c(f"│ [{i:3d}] {stripped}", C.GREY))
            try:
                process_line(stripped, state, from_script=True)
            except Exception as e:
                print(_c(f"│   错误: {e}", C.RED))
    finally:
        state.script_depth -= 1
    print(_c(f"└─ source 完成: {filepath}", C.CYAN))


def cmd_let(state, args):
    """设定变量"""
    if '=' not in args:
        print(_c("用法: let x=<值>  或  x=<值>", C.YELLOW))
        return
    name, _, value_str = args.partition('=')
    name = name.strip().lower()
    value_str = value_str.strip()
    try:
        # 尝试解析为复数 / 浮点数
        value = complex(value_str)
        if value.imag == 0:
            value = complex(value.real, 0)
    except ValueError:
        # 尝试作为数学表达式求值 (x=0)
        try:
            value = evaluate_at(value_str, complex(0), state.mode)
        except Exception:
            print(_c(f"无法解析值: {value_str}", C.RED))
            return
    state.variables[name] = value
    if name == 'x':
        state.x_value = value
    print(_c(f"  {name} = {format_complex(value)}", C.GREEN))


def cmd_vars(state, args):
    """显示变量"""
    if not state.variables:
        print(_c("  (无已定义变量)", C.GREY))
        return
    for name, value in state.variables.items():
        marker = " *" if name == 'x' else ""
        print(f"  {name} = {format_complex(value)}{marker}")


def cmd_eval(state, args):
    """在当前 x 值下求值"""
    if not args:
        print(_c("用法: eval <表达式>", C.YELLOW))
        return
    try:
        result = evaluate_at(args, state.x_value, state.mode)
        print(f"  x = {format_complex(state.x_value)}")
        print(f"  {_c('结果:', C.BOLD)} {format_complex(result)}")
    except Exception as e:
        print(_c(f"  求值错误: {e}", C.RED))


def cmd_tree(state, args):
    """切换显示 EML 解析树"""
    arg = args.strip().lower()
    if arg in ('on', '1', 'true', 'yes'):
        state.show_tree = True
    elif arg in ('off', '0', 'false', 'no'):
        state.show_tree = False
    else:
        state.show_tree = not state.show_tree
    status = _c("开", C.GREEN) if state.show_tree else _c("关", C.GREY)
    print(f"  EML 解析树显示: {status}")


def cmd_history(state, args):
    """显示历史记录"""
    if not state.history:
        print(_c("  (无历史记录)", C.GREY))
        return
    for i, line in enumerate(state.history[-50:], 1):
        print(f"  {i:3d}: {line}")


def cmd_clear(state, args):
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


# ═══════════════════════════════════════════════════════════
#  单行处理
# ═══════════════════════════════════════════════════════════

# 命令表: 命令名 → (处理函数, 需要参数)
COMMANDS = {
    'help':    (cmd_help, False),
    '?':       (cmd_help, False),
    'test':    (cmd_test, False),
    'source':  (cmd_source, True),
    'let':     (cmd_let, True),
    'vars':    (cmd_vars, False),
    'eval':    (cmd_eval, True),
    'tree':    (cmd_tree, True),
    'history': (cmd_history, False),
    'clear':   (cmd_clear, False),
    'cls':     (cmd_clear, False),
    'mode':    (None, False),  # 特殊处理
    'eml':     (None, False),
    'math':    (None, False),
}


def process_line(line: str, state: ReplState, from_script: bool = False):
    """
    处理单行输入. 返回 True 表示继续, False 表示退出.
    """
    line = line.strip()
    if not line:
        return True

    # 兼容原仓库: 行尾 \c 清空当前行
    if line.endswith("\\c"):
        return True

    # 退出命令
    if line.lower() in ("exit", "quit", "q"):
        if not from_script:
            print(_c("再见!", C.BOLD))
        return False

    # 记录历史 (非脚本模式)
    if not from_script:
        state.history.append(line)

    # 检查是否为命令
    parts = line.split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    # 特殊命令: mode / eml / math
    if cmd == "mode":
        mode_name = "EML 表达式" if state.mode == "eml" else "数学表达式"
        print(f"  当前模式: {_c(mode_name, C.CYAN)}")
        return True
    if cmd == "eml":
        state.mode = "eml"
        print(_c("  已切换到 EML 表达式输入模式", C.CYAN))
        return True
    if cmd == "math":
        state.mode = "math"
        print(_c("  已切换到数学表达式输入模式", C.CYAN))
        return True

    # 简写: x=<值> (不带 let)
    if cmd.startswith("x=") or (len(cmd) > 1 and cmd[0].lower() == 'x' and cmd[1] == '='):
        cmd_let(state, line)
        return True

    # 普通命令
    if cmd in COMMANDS:
        handler, needs_args = COMMANDS[cmd]
        if handler:
            if needs_args and not args:
                print(_c(f"用法: {cmd} <参数>", C.YELLOW))
            else:
                handler(state, args)
        return True

    # 检查 with 子句 (兼容原仓库)
    expr_str, value_expr = split_with_clause(line)
    if value_expr is not None:
        _process_with(expr_str, value_expr, state)
        return True

    # 普通表达式: 化简
    _process_expression(line, state)
    return True


def _process_expression(expr: str, state: ReplState):
    """处理普通表达式: 化简 + 可选显示解析树"""
    try:
        # 化简结果
        if state.show_simplified:
            simplified = simplify_expression(expr, state.mode)
            print(f"  {_c('化简:', C.BOLD)} {_c(simplified, C.GREEN)}")

        # EML 解析树
        if state.show_tree:
            tree = get_eml_tree(expr, state.mode)
            print(f"  {_c('EML:', C.BOLD)} {_c(tree, C.GREY)}")

    except Exception as e:
        print(_c(f"  错误: {e}", C.RED))


def _process_with(expr_str: str, value_expr: str, state: ReplState):
    """处理 with x=表达式 子句 (兼容原仓库)"""
    try:
        # 先求右侧表达式的值 (x=0)
        _, x_value = evaluate_expr(value_expr, complex(0))
        # 再求主表达式
        expr_node, result = evaluate_expr(expr_str, x_value)
        # 同时化简主表达式
        simplified = simplify_expression(expr_str, state.mode)

        print(f"  {_c('x =', C.BOLD)} {format_complex(x_value)}")
        print(f"  {_c('化简:', C.BOLD)} {_c(simplified, C.GREEN)}")
        if state.show_tree:
            print(f"  {_c('EML:', C.BOLD)} {_c(str(expr_node)[:200], C.GREY)}")
        print(f"  {_c('结果:', C.BOLD)} {format_complex(result)}")
    except Exception as e:
        print(_c(f"  错误: {e}", C.RED))


# ═══════════════════════════════════════════════════════════
#  主循环
# ═══════════════════════════════════════════════════════════

def repl(initial_mode: str = "math"):
    state = ReplState()
    state.mode = initial_mode

    print(BANNER)
    mode_name = "EML 表达式" if state.mode == "eml" else "数学表达式"
    print(f"  模式: {_c(mode_name, C.CYAN)}  |  "
          f"x = {_c(format_complex(state.x_value), C.CYAN)}  |  "
          f"输入 {_c('help', C.YELLOW)} 查看命令\n")

    while True:
        try:
            prompt = _c("eml> ", C.CYAN) if state.mode == "eml" else _c("math> ", C.GREEN)
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_c('再见!', C.BOLD)}")
            break

        if not process_line(line, state):
            break


# ═══════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]

    # --source 模式: 执行脚本文件
    if args and args[0] == "--source":
        if len(args) < 2:
            print("用法: python3 repl.py --source <文件>", file=sys.stderr)
            sys.exit(1)
        state = ReplState()
        cmd_source(state, args[1])
        return

    # --test 模式
    if args and args[0] == "--test":
        state = ReplState()
        success = cmd_test(state, "")
        sys.exit(0 if success else 1)

    # --help
    if args and args[0] in ("--help", "-h"):
        print(__doc__)
        return

    # 单次执行模式: 非 -- 开头的参数当作表达式
    if args and not args[0].startswith("--"):
        expr = " ".join(args)
        try:
            result = simplify_expression(expr, "math")
            print(result)
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # 解析标志
    mode = "math"
    for arg in args:
        if arg == "--eml":
            mode = "eml"

    # 交互模式
    repl(mode)


if __name__ == "__main__":
    main()
