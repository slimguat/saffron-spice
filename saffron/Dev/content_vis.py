from pathlib import Path
import re


class ModuleTree:
    def __init__(self, module_path,exclude=[]):
        self.exclude = exclude
        self.parent_path = Path(module_path)
        self.l ="│"
        self.c ="└"
        self.t ="├"
        self._ ="─"

    def get_pycontent(self):
        content = {}
        for path in self.parent_path.glob('*.py'):
            content[path.name] = path.read_text()
        return content
    
    def get_subtree(self,path):
      path = Path(path)
      content = {}
      for path in path.glob('*'):
        if path.name in self.exclude:
          continue
        if path.is_dir():
          content[path.name] = self.get_subtree(path)
        elif path.is_file() and path.suffix == '.py':
          content[path.name] = self.list_classes_and_functions(path)  
        else :
          content[path.name] = path.name
      return content
    
    def list_classes_and_functions(sef,file_path):
      with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
          lines = file.readlines()
      # Regular expression to match 'class' followed by any characters until a colon
      class_pattern = re.compile(r'^class\s+([^\s\(]+)\s*(\(.*\))?\s*:')
      # Regular expression to match 'def' followed by any characters until a colon
      def_pattern = re.compile(r'^def\s+([^\s\(]+)\s*\(.*\)\s*:')
      
      classes = []
      functions = []
      
      for line in lines:
          class_match = class_pattern.match(line)
          def_match = def_pattern.match(line)
          
          if class_match:
              class_name = class_match.group(1)
              inheritance = class_match.group(2) if class_match.group(2) else ""
              classes.append(f"{class_name}{inheritance}")
          
          if def_match:
              function_name = def_match.group(1)
              functions.append(function_name)
      
      return classes, functions
    
    def string_level(self,content, current_level,previous_indent=""):
      RED = "\033[31m"
      YELLOW = "\033[33m"
      GREEN = "\033[32m"
      RESET = "\033[0m"
      GREY = "\033[90m"
      line_jumps = 0
      
      indent_size = 5
      indent = " "*indent_size
      list_lines = []
      # print(content)
      for ind,(key,item) in enumerate(content.items()):
        # print(type(item))
        if isinstance(item,dict):
          
          if ind != len(content)-1:
            current_indent = f"{previous_indent}{self.t}{indent_size*self._}"
          else:
            current_indent = f"{previous_indent}{self.c}{indent_size*self._}"
          list_lines.append(f"{current_indent}{key}")
          for ijump in range(line_jumps):list_lines.append(f"{current_indent}")
          
          next_indent = f"{previous_indent}{(self.l if ind<len(content)-1 else "")}{indent_size*' '}"
          list_lines.extend(self.string_level(content[key],current_level+1,next_indent))

        elif isinstance(item,tuple):
          
          if ind != len(content)-1:
            current_indent = f"{previous_indent}{self.t}{indent_size*self._}"
          else:
            current_indent = f"{previous_indent}{self.c}{indent_size*self._}"
          list_lines.append(f"{current_indent}{YELLOW}{key}{RESET}")
          for ijump in range(line_jumps):list_lines.append(f"{current_indent}")
          
          next_indent = f"{previous_indent}{(self.l if ind<len(content)-1 else "")}{indent_size*' '}"
          
          for ind2,i in enumerate(item[0]):
            if ind2 != len(item[0])-1+len(item[1]):
              current_indent = f"{next_indent}{self.t}{indent_size*self._}"
            else:
              current_indent = f"{next_indent}{self.c}{indent_size*self._}"
              
            list_lines.append(f"{current_indent}{GREEN}Class: {i}{RESET}")
            for ijump in range(line_jumps):list_lines.append(f"{current_indent}")

          for ind2,i in enumerate(item[1]):
            if ind2 != len(item[1])-1:
              current_indent = f"{next_indent}{self.t}{indent_size*self._}"
            else:
              current_indent = f"{next_indent}{self.c}{indent_size*self._}"
            list_lines.append(f"{current_indent}{RED}Function: {i}{RESET}")
            for ijump in range(line_jumps):list_lines.append(f"{current_indent}")
            
        else:
          if ind != len(content)-1:
            current_indent = f"{previous_indent}{self.t}{indent_size*self._}"
          else:
            current_indent = f"{previous_indent}{self.c}{indent_size*self._}"
          list_lines.append(f"{current_indent}{GREY}{key}{RESET}")
          for ijump in range(line_jumps):list_lines.append(f"{current_indent}")

      return list_lines
      
    def __repr__(self) -> str:
      
      content = self.get_subtree(self.parent_path)
      list_lines= self.string_level(content,0)
      return (f"{self.parent_path}\n"+"\n".join(list_lines))
      # return ("\n".join(list_lines))


def module_content(module_path,exclude=["__pycache__",".git"]):
  print(ModuleTree(module_path,exclude))