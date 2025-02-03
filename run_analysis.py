import json
from pathlib import Path
import libcst as cst
import ast
import re, os, csv
from hityper.typeobject import TypeObject
import shutil
import subprocess
from typet5.type_check import parse_type_str

def match_type_for_cot(string):
    pattern = re.compile(r'\`[a-zA-Z\.]+(?:\[[a-zA-Z\. ]+(?:\,[a-zA-Z\. ]+)*\])*\`')
    matched = re.findall(pattern, string)
    if len(matched) == 0:
        second_pattern = re.compile(r'\`[a-zA-Z\.\,\[\] ]+\`')
        second_matched = re.findall(second_pattern, string)
        if len(second_matched) == 0:
            return None
        else:
            res = second_matched[-1].replace("`", "").replace('NoneType', 'None')#.replace("is ", "")
            if (" " in res and "[" not in res) or res.lower() == "unknown":
                res = None
            return res
    else:
        res = matched[-1].replace("`", "").replace('NoneType', 'None')#.replace("is ", "")
        if (" " in res and "[" not in res) or res.lower() == "unknown":
            res = None
        return res

def match_type(string):
    string = string.split("\nPython Code:")[0].split("\nQ:")[0]
    pattern = re.compile(r'\`[a-zA-Z\.]+(?:\[[a-zA-Z\. ]+(?:\,[a-zA-Z\. ]+)*\])*\`')
    matched = re.findall(pattern, string)
    if len(matched) == 0:
        second_pattern = re.compile(r'\`[a-zA-Z\.\,\[\] ]+\`')
        second_matched = re.findall(second_pattern, string)
        if len(second_matched) == 0:
            return string.split("\n")[0][:-1]
        else:
            return second_matched[0].replace("`", "")
    else:
        return matched[0].replace("`", "")

def match_type_for_completion(string):
    string = string.split("\nPython Code:")[0].split("\nQ:")[0]
    pattern = re.compile(r'[a-zA-Z\.]+(?:\[[a-zA-Z\. ]+(?:\,[a-zA-Z\. ]+)*\])*')
    matched = re.findall(pattern, string)
    if len(matched) == 0:
        second_pattern = re.compile(r'[a-zA-Z\.\,\[\] ]+')
        second_matched = re.findall(second_pattern, string)
        if len(second_matched) == 0:
            return string.split("\n")[0][:-1]
        else:
            return second_matched[0].replace("`", "")
    else:
        return matched[0].replace("`", "")


def extract_type_from_text(text):
    if len(text.split()) > 0:
        text = text.split()[0]
    else:
        text = text
    if text.endswith(".") or text.endswith(","):
        text = text[:-1]
    typeobjs = TypeObject.Str2Obj(text)
    return typeobjs


def extract_type_from_cot(text):
    text = text.split()[-1][:-1]
    typeobjs = TypeObject.Str2Obj(text)
    return typeobjs


class FunctionLocator(ast.NodeVisitor):
    def __init__(self):
        self.inclass = False
        self.inclass = False
        self.found = False
        self.node = None


    def visit_ClassDef(self, node):
        if not self.inclass and node.name == self.classname:
            self.inclass = True
            self.found = False
            self.generic_visit(node)
            if self.found and self.funcname == "global":
                self.node = node
        elif not self.inclass:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_FunctionDef(self, node):
        if not self.infunc and node.name == self.funcname and self.inclass:
            if self.scope == 'return' and node.name == self.name:
                self.node = node
            else:
                self.infunc = True
                self.found = False
                self.generic_visit(node)
                if self.found:
                    self.node = node
        elif not self.infunc and self.inclass:
            self.generic_visit(node)
    
    def visit_Name(self, node):
        if node.id == self.name and self.scope == 'local' and self.infunc and self.inclass:
            self.found = True
    
    def visit_Attribute(self, node):
        if node.attr == self.name and hasattr(node.value, "id") and node.value.id == "self" and self.scope == "local" and self.infunc and self.inclass:
            self.found = True

    def visit_arg(self, node):
        if node.arg == self.name and self.scope == 'arg' and self.infunc and self.inclass:
            self.found = True


    def run(self, root, loc, name, scope):
        self.inclass = False
        self.infunc = False
        self.node = None
        self.found = False
        self.funcname, self.classname = loc.split('@')
        self.name = name
        self.scope = scope
        if self.classname == 'global':
            self.inclass = True
        if self.funcname == 'global':
            self.infunc = True
        if self.inclass and self.infunc:
            remover = GlobalNodeRemover()
            node = remover.run(root)
            return node
        else:
            self.visit(root)
        return self.node

def is_kwonlyargs(node, var_name, is_param):
    if is_param:
        for arg in node.args.kwonlyargs:
            if arg.arg == var_name:
                return True
    return False

def annotate_function(func_node: ast.FunctionDef, typ: str, var_name, is_param=True, is_removed=False) -> ast.FunctionDef:
    # 파라미터에 타입 추가
    if is_removed:
        if is_param:
            for arg in func_node.args.posonlyargs:
                if arg.arg == var_name:
                    arg.annotation = None
                    break
            for arg in func_node.args.args:
                if arg.arg == var_name:
                    arg.annotation = None
                    break
            if func_node.args.vararg and func_node.args.vararg.arg == var_name:
                func_node.args.vararg.annotation = None
            elif func_node.args.kwarg and func_node.args.kwarg.arg == var_name:
                func_node.args.kwarg.annotation = None
            elif func_node.args.kwonlyargs:
                for arg in func_node.args.kwonlyargs:
                    if arg.arg == var_name:
                        arg.annotation = None
                        break
        # 리턴 타입 추가
        else:
            func_node.returns = None
    else:
        if is_param:
            for arg in func_node.args.posonlyargs:
                if arg.arg == var_name:
                    arg.annotation = ast.Name(id=typ, ctx=ast.Load())
                    break
            for arg in func_node.args.args:
                if arg.arg == var_name:
                    arg.annotation = ast.Name(id=typ, ctx=ast.Load())
                    break
            if func_node.args.vararg and func_node.args.vararg.arg == var_name:
                func_node.args.vararg.annotation = ast.Name(id=typ, ctx=ast.Load())
            elif func_node.args.kwarg and func_node.args.kwarg.arg == var_name:
                func_node.args.kwarg.annotation = ast.Name(id=typ, ctx=ast.Load())
            elif func_node.args.kwonlyargs:
                for arg in func_node.args.kwonlyargs:
                    if arg.arg == var_name:
                        arg.annotation = ast.Name(id=typ, ctx=ast.Load())
                        break
        # 리턴 타입 추가
        else:
            func_node.returns = ast.Name(id=typ, ctx=ast.Load())
    
    return func_node

class OverrideAnnotationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)  # ParentNodeProvider 추가

    def __init__(self, target_name: str, annotations: dict, return_annotation: str):
        # 타겟 함수 이름, 인수 주석 및 반환 주석을 초기화
        if "." in target_name:
            splited = target_name.split(".")
            self.target_class, self.target_function = splited[:-1], splited[-1]
        else:
            self.target_class = None
            self.target_function = target_name
        self.annotations = annotations
        self.return_annotation = return_annotation

    def _is_in_target_class(self, node: cst.FunctionDef) -> bool:
        # 주어진 함수 노드가 target_class 안에 있는지 확인하는 내부 함수
        current_node = self.get_metadata(cst.metadata.ParentNodeProvider, node)
        check_class_pos = len(self.target_class) - 1

        while current_node:
            if isinstance(current_node, cst.ClassDef):
                if current_node.name.value == self.target_class[check_class_pos]:
                    if check_class_pos == 0:
                        return True
                    else:
                        check_class_pos -= 1
                else:
                    return False
            elif isinstance(current_node, cst.Module):
                return False
            current_node = self.get_metadata(cst.metadata.ParentNodeProvider, current_node)
        return False

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        # 클래스 안의 메서드일 경우
        if self.target_class:
            # 현재 함수가 target_class 내부에 있는지 확인
            if not self._is_in_target_class(original_node):
                return updated_node

        # 함수 이름이 타겟 함수와 일치할 경우에만 주석을 덮어씌움
        if original_node.name.value == self.target_function:
            # 각 파라미터에 대해 새로운 주석 설정
            new_params = []
            for param in updated_node.params.params:
                param_name = param.name.value
                if param_name in self.annotations:
                    # 새 주석 적용
                    new_annotation = self.annotations[param_name]
                    new_param = param.with_changes(annotation=new_annotation)
                else:
                    # 해당하는 주석이 없으면 그대로 유지
                    new_param = param.with_changes(annotation=None)
                new_params.append(new_param)

            # 반환 타입 주석 덮어쓰기
            new_return_annotation = self.return_annotation

            return updated_node.with_changes(
                params=updated_node.params.with_changes(params=new_params),
                returns=new_return_annotation
            )
        return updated_node

class RemoveAnnotationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)  # ParentNodeProvider 추가

    def __init__(self, target_name: str, annotations: dict, return_annotation: str):
        # 타겟 함수 이름, 인수 주석 및 반환 주석을 초기화
        if "." in target_name:
            splited = target_name.split(".")
            self.target_class, self.target_function = splited[:-1], splited[-1]
        else:
            self.target_class = None
            self.target_function = target_name
        self.annotations = annotations
        self.return_annotation = return_annotation

    def _is_in_target_class(self, node: cst.FunctionDef) -> bool:
        # 주어진 함수 노드가 target_class 안에 있는지 확인하는 내부 함수
        current_node = self.get_metadata(cst.metadata.ParentNodeProvider, node)
        check_class_pos = len(self.target_class) - 1

        while current_node:
            if isinstance(current_node, cst.ClassDef):
                if current_node.name.value == self.target_class[check_class_pos]:
                    if check_class_pos == 0:
                        return True
                    else:
                        check_class_pos -= 1
                else:
                    return False
            elif isinstance(current_node, cst.Module):
                return False
            current_node = self.get_metadata(cst.metadata.ParentNodeProvider, current_node)
        return False

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        # 클래스 안의 메서드일 경우
        if self.target_class:
            # 현재 함수가 target_class 내부에 있는지 확인
            if not self._is_in_target_class(original_node):
                return updated_node

        # 함수 이름이 타겟 함수와 일치할 경우에만 주석을 덮어씌움
        if original_node.name.value == self.target_function:
            # 각 파라미터 주석 제거
            new_params = []
            for param in updated_node.params.params:
                new_param = param.with_changes(annotation=None)
                new_params.append(new_param)

            # 반환 타입 주석 제거
            new_return_annotation = None

            return updated_node.with_changes(
                params=updated_node.params.with_changes(params=new_params),
                returns=new_return_annotation
            )
        return updated_node


class VarAnnotationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)  # ParentNodeProvider 추가

    def __init__(self, target_name, original_annot, target_annot, in_class):
        if "." in target_name:
            splited = target_name.split(".")
            self.target_class, self.target_var = splited[:-1], splited[-1]
        else:
            self.target_class = None
            self.target_var = target_name
        self.original_annot = original_annot
        self.target_annot = target_annot
        self.in_class = in_class

    def leave_AnnAssign(self, original_node, updated_node):
        # 변수에 달린 annotation을 변경
        if updated_node.target.value == self.target_var and self._has_original_annotation(updated_node.annotation):
            return updated_node.with_changes(annotation=self.target_annot)
        elif self.in_class:
            if isinstance(updated_node.target, cst.Attribute):
                temp_module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=updated_node.target)])])
                if temp_module.code.startswith("self."):
                    if updated_node.target.attr == self.target_var and self._has_original_annotation(updated_node.annotation):
                        return updated_node.with_changes(annotation=self.target_annot)

        return updated_node

    def _has_original_annotation(self, annotation):
        # annotation이 original_annot인지 확인
        return parse_type_expr(annotation.annotation) == parse_type_expr(self.original_annot.annotation)

class RemoveVarAnnotationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)  # ParentNodeProvider 추가

    def __init__(self, target_name, original_annot, in_class):
        if "." in target_name:
            splited = target_name.split(".")
            self.target_class, self.target_var = splited[:-1], splited[-1]
        else:
            self.target_class = None
            self.target_var = target_name
        self.original_annot = original_annot
        self.in_class = in_class


    def leave_AnnAssign(self, original_node, updated_node):
        # 변수에 달린 annotation을 변경
        if updated_node.target.value == self.target_var and self._has_original_annotation(updated_node.annotation):
            if updated_node.value is None:
                return cst.SimpleStatementLine(
                    body=[cst.Expr(value=updated_node.target)]
                )
            # Otherwise, transform normally
            return cst.Assign(
                targets=[cst.AssignTarget(target=updated_node.target)],
                value=updated_node.value,
            )
        elif self.in_class:
            if isinstance(updated_node.target, cst.Attribute) and parse_type_expr(updated_node.target.value).startswith("self."):
                if updated_node.target.attr == self.target_var and self._has_original_annotation(updated_node.annotation):
                    if updated_node.value is None:
                        return cst.SimpleStatementLine(
                            body=[cst.Expr(value=updated_node.target)]
                        )
                    # Otherwise, transform normally
                    return cst.Assign(
                        targets=[cst.AssignTarget(target=updated_node.target)],
                        value=updated_node.value,
                    )

        return updated_node
    
    def _has_original_annotation(self, annotation):
        # annotation이 original_annot인지 확인
        return parse_type_expr(annotation.annotation) == parse_type_expr(self.original_annot.annotation)


def run():
    typegen_path = Path.home() / "TypeGen"
    pred_path = Path("data/predictions")
    testset_path = Path("data/new_testset.json")

    with open(pred_path / "typegen.json", "r") as f:
        typegen_result = json.load(f)

    with open(testset_path, "r") as f:
        testset = json.load(f)

    kind_set = set() 

    for repo, msgs in typegen_result.items():
        result_path = Path("analysis_result") / repo.replace("/", "+")

        file_path = repo[:repo.find(".py--")+3]
        proj_path = "/".join(file_path.split("/")[:3])
        content = repo.split(".py--")[1]

        try:
            content_split = content.split("--")
            contexts = content_split[0]
            var_name = content_split[1]
            kind = content_split[2]

            contexts_split = contexts.split("@")
            method_name = contexts_split[0]
            class_name = contexts_split[1]

            if kind in ["arg", "return"]:
                print(f"Processing {repo}")

                if not result_path.exists():
                    os.makedirs(result_path)
                
                predictions = []
                if len(msgs) == 0:
                    print("Empty")
                    continue

                for res in msgs[:10]:
                    pred = match_type_for_cot(res.split('\nPython')[0])
                    if pred == None:
                        try:
                            predictions.append(res.split('\nPython')[0].split()[-1][:-1] if res.split('\nPython')[0].split()[-1].endswith(".") else res.split('\nPython')[0].split()[-1])
                        except:
                            predictions.append("invalid_type")
                    else:
                        predictions.append(pred)

                assert len(predictions) == 10

                code_annotation = None
                for test in testset:
                    if (test["file"] == file_path and \
                        test["loc"] == f"{method_name}@{class_name}" and \
                        test["name"] == var_name and \
                        test["scope"] == kind):
                        code_annotation = test["code_annotation"]
                        break

                if code_annotation is None:
                    print("No code annotation")
                    continue

                correct_idx = []
                incorrect_idx = []

                for index, pred in enumerate(predictions):
                    try:
                        parse_predtype = parse_type_str(pred).normalized()
                        parse_gttype = parse_type_str(code_annotation).normalized()

                        if parse_predtype == parse_gttype:
                            correct_idx.append(index)
                        else:
                            incorrect_idx.append(index)
                    except:
                        incorrect_idx.append(index)

                
                # save info
                # for index, pred in enumerate(predictions):
                #     try:
                #         parse_predtype = PythonType.from_str(str(pred)).normalized()
                #     except:
                #         print(str(pred))
                #         predtype = TypeObject.Str2Obj(pred)
                #         if predtype:
                #             print(predtype[0])
                #         input()
                #         continue
                #     print(str(parse_predtype))
                #     continue
                #     predtype = TypeObject.Str2Obj(pred)

                #     if TypeObject.isIdenticalSet(gttype, predtype):
                #         correct_idx.append(index)
                #     else:
                #         incorrect_idx.append(index)

                # continue

                info = {
                    'sig_name': repo,
                    'correct': correct_idx,
                    'incorrect': incorrect_idx,
                }

                with open(result_path / "info.json", "w") as f:
                    json.dump(info, f, indent=4)

                file_info = {
                    'file_path' : file_path,
                    'sig_name' : contexts,
                    'var_name' : var_name,
                }

                with open(result_path / "file_info.json", "w") as f:
                    json.dump(file_info, f, indent=4)

                # analysis correct code
                try:
                    # copy original file
                    shutil.copyfile(file_path, file_path + ".bak")

                    if True or not os.path.exists(result_path / "correct.json"):
                        # read original file and add correct annotation
                        with open(file_path, "r") as f:
                            code = f.read()

                        root = ast.parse(code)
                        locator = FunctionLocator()
                        node = locator.run(root, f"{method_name}@{class_name}", var_name, kind)

                        if node is None:
                            continue

                        func_node = annotate_function(node, str(code_annotation), var_name, kind == "arg")

                        with open(file_path, "w") as f:
                            f.write(ast.unparse(root))
                            
                        print("Analysis Correct Code")

                        p = subprocess.Popen(
                            ['pyright', '--outputjson', typegen_path / file_path], 
                            stdout=subprocess.PIPE,
                            cwd=typegen_path / proj_path
                        )
                        output, _ = p.communicate()
                        output_json = output.decode("utf-8")

                        # save original analysis result
                        with open(result_path / "correct.json", 'w') as f:
                            f.write(output_json)
                finally:
                    # restore original file
                    if os.path.exists(file_path + ".bak"):
                        shutil.move(file_path + ".bak", file_path)
                

                for index, pred in enumerate(predictions):
                    try:
                        parse_predtype = parse_type_str(str(pred)).normalized()
                    except:
                        continue
                    # predtype = TypeObject.Str2Obj(pred)

                    # if TypeObject.isIdenticalSet(gttype, predtype):
                    #     continue

                    try:
                        # copy original file
                        shutil.copyfile(file_path, file_path + ".bak")

                        is_flag = is_kwonlyargs(func_node, var_name, kind == "arg")
                        if is_flag or not os.path.exists(result_path / f"modified_{index}.json"):
                            # read original file and add correct annotation
                            with open(file_path, "r") as f:
                                code = f.read()

                            root = ast.parse(code)
                            locator = FunctionLocator()
                            node = locator.run(root, f"{method_name}@{class_name}", var_name, kind)

                            if node is None:
                                continue

                            func_node = annotate_function(node, str(parse_predtype), var_name, kind == "arg")

                            with open(file_path, "w") as f:
                                f.write(ast.unparse(root))
                                
                            print(f"Analysis {index}th Code")

                            p = subprocess.Popen(
                                ['pyright', '--outputjson', typegen_path / file_path], 
                                stdout=subprocess.PIPE,
                                cwd=typegen_path / proj_path
                            )
                            output, _ = p.communicate()
                            output_json = output.decode("utf-8")

                            # save original analysis result
                            with open(result_path / f"modified_{index}.json", 'w') as f:
                                f.write(output_json)
                    finally:
                        # restore original file
                        if os.path.exists(file_path + ".bak"):
                            shutil.move(file_path + ".bak", file_path)

                # analysis removed code
                try:
                    # copy original file
                    shutil.copyfile(file_path, file_path + ".bak")

                    is_flag = is_kwonlyargs(func_node, var_name, kind == "arg")
                    if is_flag or not os.path.exists(result_path / "removed.json"):
                        # read original file and add correct annotation
                        with open(file_path, "r") as f:
                            code = f.read()

                        root = ast.parse(code)
                        locator = FunctionLocator()
                        node = locator.run(root, f"{method_name}@{class_name}", var_name, kind)

                        if node is None:
                            continue

                        func_node = annotate_function(node, None, var_name, kind == "arg", True)

                        with open(file_path, "w") as f:
                            f.write(ast.unparse(root))

                        print("Analysis Removed Code")

                        p = subprocess.Popen(
                            ['pyright', '--outputjson', typegen_path / file_path], 
                            stdout=subprocess.PIPE,
                            cwd=typegen_path / proj_path
                        )
                        output, _ = p.communicate()
                        output_json = output.decode("utf-8")

                        # save original analysis result
                        with open(result_path / "removed.json", 'w') as f:
                            f.write(output_json)

                finally:
                    # restore original file
                    if os.path.exists(file_path + ".bak"):
                        shutil.move(file_path + ".bak", file_path)

        except FileNotFoundError:
            continue


if __name__ == "__main__":
    run()