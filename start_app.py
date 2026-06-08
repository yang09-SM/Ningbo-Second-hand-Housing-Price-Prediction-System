#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

# 设置工作目录为项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, project_root)

print(f"工作目录: {os.getcwd()}")
print(f"Python路径: {sys.path[:3]}")

# 现在导入并运行Flask应用
from src.web.app import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
