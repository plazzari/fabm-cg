#!/usr/bin/env python

import ast
import argparse
from typing import Mapping, Tuple, Iterable, Type, Optional
from types import ModuleType
import os
import importlib.util
import collections
import sys

import numpy as np

import ingredients


FORTRAN_TYPES = {float: "real(rk)", int: "integer", bool: "logical"}
BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
UNARYOPS = {ast.UAdd: "+", ast.USub: "-"}
CMPOPS = {
    ast.Eq: "==",
    ast.NotEq: "/=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Lt: "<",
    ast.LtE: "<=",
}
NUMPY_UFUNCS = {
    np.exp: "exp",
    np.tanh: "tanh",
    np.log10: "log10",
    np.log: "log",
    np.sqrt: "sqrt",
}


def translate(path: str, outpath: Optional[str] = None):
    # Load the module
    module_name = os.path.basename(os.path.splitext(path)[0])
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    # Get abstract syntax tree
    with open(path) as f:
        tree = ast.parse(f.read(), type_comments=True)

    # Translate each AST node
    vis = FABMTranslator(module)
    translation = vis.visit(tree)

    # Write final translation to file or stdout
    out = sys.stdout if outpath is None else open(outpath, "w")
    translation.write(out)
    if outpath is not None:
        out.close()


def infer_type(node: ast.AST, default: Type) -> Type:
    if isinstance(node, ast.Name):
        return eval(node.id)
    return default


class Block(collections.UserList):
    def __init__(self, items: Iterable = (), separate: bool = False):
        super().__init__(items)
        self.separate = separate

    def write(self, f, depth: int = 0):
        if self.separate and self.data:
            f.write("\n")
        previous_at_top = False
        for item in self.data:
            if isinstance(item, Block):
                item.write(f, depth + 1)
                previous_at_top = False
            else:
                if previous_at_top and self.separate:
                    f.write("\n")
                f.write("  " * depth + item + "\n")
                previous_at_top = True
        if self.separate and self.data:
            f.write("\n")


class Visitor(ast.NodeVisitor):
    def __init__(self, module: ModuleType):
        self.functions = set(["min", "max"])
        self.globals = set()
        self.locals = collections.OrderedDict()
        self.module = module

        # context when inside a class or function
        self.cls: Optional[Type[ingredients.Model]] = None
        self.return_name: Optional[str] = None
        self.self_name: Optional[str] = None
        self.readable = {}
        self.read = collections.OrderedDict()

    def visit_Module(self, node: ast.Module) -> Block:
        # Collect globals
        for child in node.body:
            if isinstance(child, ast.Assign):
                # global variable
                assert len(child.targets) == 1 and isinstance(
                    child.targets[0], ast.Name
                )
                self.globals.add(child.targets[0].id)
            elif isinstance(child, ast.FunctionDef):
                # global function
                self.functions.add(child.name)

        classes, functions = [], []
        for n in node.body:
            if isinstance(n, ast.ClassDef):
                classes.append(self.visit(n))
            elif isinstance(n, ast.FunctionDef):
                functions.append(self.visit(n))
        return self.translate_module(self.module.__name__, classes, functions)

    def visit_ClassDef(self, node: ast.ClassDef) -> Tuple[Block, Block]:
        assert self.cls is None, "nested classes not supported"
        cls: Type = getattr(self.module, node.name)
        assert issubclass(cls, ingredients.Model)
        self.cls = cls
        self.readable.update(self.cls.state_variables)
        self.readable.update(self.cls.dependencies)
        self.readable.update(self.cls.parameters)

        funcs = collections.OrderedDict()
        for n in node.body:
            if isinstance(n, ast.FunctionDef):
                funcs[n.name] = self.visit(n)

        res = self.translate_class(node.name, self.cls, funcs)

        self.cls = None
        self.readable.clear()

        return res

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Block:
        self.locals.clear()
        if self.cls:
            # class method
            self.self_name = node.args.args[0].arg
            self.read.clear()
        else:
            # regular [unbound] function
            for a in node.args.args:
                self.locals[a.arg] = True
            self.return_name = node.name

        body_block = Block()
        for n in node.body:
            body_block.extend(self.visit(n))

        if self.cls:
            # class method
            return self.translate_method(
                node.name, body_block, self.cls, self.read, self.locals
            )
        else:
            # regular [unbound] function
            args = collections.OrderedDict()
            for a in node.args.args:
                args[a.arg] = infer_type(a.annotation, float)
            return_type = infer_type(node.returns, float)
            locals = [loc for loc in self.locals if loc not in args]
            return self.translate_function(
                node.name, body_block, args, return_type, locals
            )

    def visit_If(self, node: ast.If) -> Block:
        block = Block(["if (%s) then" % self.visit(node.test)])
        for n in node.body:
            block.append(self.visit(n))
        if node.orelse:
            block.append("else")
            for n in node.orelse:
                block.append(self.visit(n))
        block.append("end if")
        return block

    def visit_Return(self, node: ast.Return) -> Block:
        assert self.return_name
        return Block(["%s = %s" % (self.return_name, self.visit(node.value)), "return"])

    def visit_Assign(self, node: ast.Assign) -> Block:
        assert len(node.targets) == 1, "assign to multipe targets not supported"
        target = node.targets[0]
        if isinstance(target, ast.Name):
            # assign to local name
            self.locals[target.id] = True
            return Block(["%s = %s" % (self.visit(target), self.visit(node.value))])
        elif isinstance(target, ast.Attribute):
            # assign to attribute - typically a diagnostic variable
            assert self.cls
            self.ensure_self(target.value)
            assert isinstance(target, ast.Attribute)
            diag = self.cls.diagnostic_variables.get(target.attr)
            assert diag, (
                "%s cannot be assigned to because it is not a diagnostic variable"
                % target.attr
            )
            return Block(
                [self.translate_diagnostic_assignment(diag, self.visit(node.value))]
            )

    def visit_AugAssign(self, node: ast.AugAssign) -> Block:
        assert self.cls
        assert isinstance(node.target, ast.Attribute)
        assert node.target.attr in ("source", "bottom_flux", "surface_flux")
        owner = node.target.value
        assert isinstance(owner, ast.Attribute)
        statevar = self.cls.state_variables[owner.attr]
        assert (
            isinstance(statevar, ingredients.InteriorStateVariable)
            or node.target.attr == "source"
        )
        assert statevar is not None
        self.ensure_self(owner.value)
        assert isinstance(node.op, (ast.Add, ast.Sub))
        value = self.visit(node.value)
        if isinstance(node.op, ast.Sub):
            value = "-" + value
        return Block(
            [self.translate_source_increment(statevar, value, node.target.attr)]
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        self.ensure_local(node.operand)
        return "(%s%s)" % (UNARYOPS[type(node.op)], self.visit(node.operand))

    def visit_BinOp(self, node: ast.BinOp) -> str:
        self.ensure_local(node.left)
        self.ensure_local(node.right)
        return "(%s %s %s)" % (
            self.visit(node.left),
            BINOPS[type(node.op)],
            self.visit(node.right),
        )

    def visit_Compare(self, node: ast.Compare) -> str:
        assert len(node.ops) == 1 and len(node.comparators) == 1
        return "%s %s %s" % (
            self.visit(node.left),
            CMPOPS[type(node.ops[0])],
            self.visit(node.comparators[0]),
        )

    def visit_Attribute(self, node: ast.Attribute) -> str:
        if isinstance(node.ctx, ast.Load):
            if getattr(node, "called", False):
                assert isinstance(node.value, ast.Name)
                m = getattr(self.module, node.value.id)
                f = getattr(m, node.attr)
                return NUMPY_UFUNCS[f]
            else:
                self.ensure_self(node.value)
                obj = self.readable.get(node.attr)
                assert obj, "Do not know how to load %s (line %i)" % (
                    node.attr,
                    node.lineno,
                )
                if not isinstance(obj, ingredients.Parameter):
                    self.read[obj] = True
                    return node.attr
        return f"{self.visit(node.value)}%{node.attr}"

    def visit_Name(self, node: ast.Name) -> str:
        if getattr(node, "called", False):
            assert node.id in self.functions, "Unknown function %s called" % node.id
        return node.id

    def visit_Constant(self, node: ast.Constant) -> str:
        return self.translate_constant(node.value)

    def visit_Call(self, node: ast.Call) -> str:
        node.func.called = True
        return "%s(%s)" % (
            self.visit(node.func),
            ", ".join([self.visit(n) for n in node.args]),
        )

    def ensure_local(self, node: ast.AST):
        assert (
            not isinstance(node, ast.Name)
            or node.id in self.globals
            or node.id in self.locals
        ), ("Unknown local: %s" % node.id)

    def ensure_self(self, node: ast.AST):
        assert isinstance(node, ast.Name) and node.id == self.self_name


class FABMTranslator(Visitor):
    def translate_module(
        self,
        name: str,
        classes: Iterable[Tuple[Block, Block]],
        functions: Iterable[Block],
    ) -> Block:
        declarations = Block(
            ["use fabm_types", "implicit none", "private"], separate=True
        )
        contains = Block(separate=True)
        for class_declarations, class_body in classes:
            declarations.extend(class_declarations)
            contains.extend(class_body)
        for function_body in functions:
            contains.extend(function_body)
        return Block(
            [
                '#include "fabm_driver.h"',
                f"module {name}",
                declarations,
                "contains",
                contains,
                "end module",
            ],
            separate=True,
        )

    def translate_class(
        self, name: str, cls: ingredients.Model, funcs: Mapping[str, Block]
    ):
        type_contains = Block(["procedure :: initialize"])
        type_contains.extend(f"procedure :: {fname}" for fname in funcs)

        func_block = Block()
        for fnbody in funcs.values():
            func_block.extend(fnbody)

        # Translate to Fortran/FABM
        type_members = Block()
        initialize_body = Block([f"class(type_{name}), intent(inout) :: self"])
        for vname, var in cls.state_variables.items():
            type_members.append(f"type({var.id_type}) :: id_{vname}")
            initialize_body.append(
                f"call self%register_state_variable("
                f"self%id_{vname}, '{vname}', '{var.units}', '{var.long_name}'"
                ")"
            )
        for vname, var in cls.diagnostic_variables.items():
            type_members.append(f"type({var.id_type}) :: id_{vname}")
            initialize_body.append(
                f"call self%register_diagnostic_variable("
                f"self%id_{vname}, '{vname}', '{var.units}', '{var.long_name}'"
                ")"
            )
        for vname, var in cls.dependencies.items():
            type_members.append(f"type({var.id_type}) :: id_{vname}")
            initialize_body.append(
                f"call self%register_dependency("
                f"self%id_{vname}, '{vname}', '{var.units}', '{var.long_name}'"
                ")"
            )
        for vname, var in cls.parameters.items():
            type_members.append("%s :: %s" % (FORTRAN_TYPES[var.type], vname))
            default = self.translate_constant(var.default)
            initialize_body.append(
                f"call self%get_parameter("
                f"self%{vname}, '{vname}', '{var.units}', '{var.long_name}'"
                f", default={default}"
                ")"
            )
        declaration = Block(
            [
                f"type, extends(type_base_model), public :: type_{name}",
                type_members,
                "contains",
                type_contains,
                "end type",
            ]
        )
        block = Block(
            [
                "subroutine initialize(self)",
                initialize_body,
                "end subroutine initialize",
            ]
            + func_block
        )
        return declaration, block

    def translate_function(
        self,
        name: str,
        body_block: Block,
        args: Mapping[str, Type],
        return_type: Type,
        locals: Iterable[str],
    ):
        arg_block = Block()
        for argname, argtype in args.items():
            arg_block.append(f"{FORTRAN_TYPES[argtype]}, intent(in) :: {argname}")
        arg_block.extend(f"real(rk) :: {locname}" for locname in locals)
        strreturn_type = FORTRAN_TYPES[return_type]
        strargs = {", ".join(args)}
        return Block(
            [
                f"elemental {strreturn_type} function {name}({strargs})",
                arg_block,
                body_block,
                f"end function {name}",
            ]
        )

    def translate_method(
        self,
        name: str,
        body_block: Block,
        cls: Type[ingredients.Model],
        inputs: Iterable[ingredients.Base],
        locals: Iterable[str],
    ):
        context = {"do_bottom": "bottom", "do_surface": "surface"}.get(name, "interior")
        args, loop = {
            "interior": ("_ARGUMENTS_DO_", "_LOOP_"),
            "bottom": ("_ARGUMENTS_DO_BOTTOM_", "_BOTTOM_LOOP_"),
            "surface": ("_ARGUMENTS_DO_SURFACE_", "_SURFACE_LOOP_"),
        }[context]
        arg_block = Block()
        arg_block.append(f"class (type_{cls.__name__}), intent(in) :: self")
        arg_block.append(f"_DECLARE{args}")
        arg_block.extend(f"real(rk) :: {v.name}" for v in inputs)
        arg_block.extend(f"real(rk) :: {locname}" for locname in locals)
        get_block = Block()
        for v in inputs:
            get_block.append(f"{v.get_macro}(self%id_{v.name}, {v.name})")
        return Block(
            [
                f"subroutine {name}(self, {args})",
                arg_block,
                Block(
                    [
                        f"{loop}BEGIN_",
                        get_block,
                        body_block,
                        f"{loop}END_",
                    ]
                ),
                f"end subroutine {name}",
            ]
        )

    def translate_constant(self, value):
        if isinstance(value, float):
            return f"{value}_rk"
        elif isinstance(value, bool):
            return ".true." if value else ".false."
        else:
            return repr(value)

    def translate_diagnostic_assignment(
        self, var: ingredients.BaseDiagnosticVariable, value
    ):
        return f"{var.set_macro}(self%id_{var.name}, {value})"

    def translate_source_increment(
        self, var: ingredients.StateVariable, value, type: str = "source"
    ):
        return f"{var.add_source_macros[type]}(self%id_{var.name}, {value})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Python file to convert to FABM/Fortran")
    parser.add_argument("out", help="Fortran file to write to (default to stdout)")
    args = parser.parse_args()
    translate(args.file, args.out)
