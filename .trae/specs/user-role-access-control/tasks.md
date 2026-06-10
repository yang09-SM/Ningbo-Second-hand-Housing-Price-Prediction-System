# Tasks

- [x] Task 1: 数据库新增 users 表并初始化默认管理员账户
  - [x] 修改 `src/database/init_db.py`，新增 users 表（id, username, password, role 字段）
  - [x] 初始化时插入默认管理员账户（admin / admin123，角色为 admin）
  - [x] 执行数据库初始化脚本

- [x] Task 2: 后端用户认证与权限控制
  - [x] 在 `src/web/app.py` 中新增登录/登出 API（`POST /api/login`, `POST /api/logout`, `GET /api/current-user`）
  - [x] 实现基于 Session 的认证机制
  - [x] 实现权限装饰器 `@admin_required`，用于保护写操作 API
  - [x] 对 `POST /api/houses`、`PUT /api/houses/<id>`、`DELETE /api/houses/<id>` 添加权限校验

- [x] Task 3: 前端登录 UI 与角色感知界面
  - [x] 在导航栏新增登录/登出按钮和当前用户信息显示区域
  - [x] 新增登录弹窗模态框（用户名 + 密码输入）
  - [x] 根据用户角色条件渲染"新增房源"按钮（仅管理员可见）
  - [x] 根据用户角色条件渲染每个房源卡片的"编辑"和"删除"按钮（仅管理员可见）
  - [x] 未登录状态以游客身份运行（等同于普通用户权限）

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 2]
