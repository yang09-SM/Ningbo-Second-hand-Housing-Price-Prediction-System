# 用户角色权限管理系统 - Verification Checklist

## 数据库 users 表
- [x] users 表创建成功，包含 id、username、password、role 字段
- [x] 默认管理员账户（admin/admin123）初始化成功
- [x] 可通过 SQL 查询验证用户数据

## 后端认证与权限控制
- [x] POST /api/login 接口正常工作，正确返回用户信息和角色
- [x] POST /api/logout 接口正常工作，正确清除会话
- [x] GET /api/current_user 接口正常返回当前登录用户信息（未登录时返回 null）
- [x] 普通用户/游客调用 POST/PUT/DELETE 房源 API 返回 403
- [x] 管理员调用 POST/PUT/DELETE 房源 API 正常执行
- [x] 只读接口（GET 列表、详情、预测等）对所有角色可访问

## 前端角色感知 UI
- [x] 导航栏显示登录按钮（未登录时）或 用户名+登出按钮（已登录时）
- [x] 登录弹窗可正常弹出，输入凭据后可完成登录
- [x] 未登录/普通用户状态下不显示"新增房源"按钮
- [x] 未登录/普通用户状态下房源卡片不显示"编辑"和"删除"按钮
- [x] 管理员登录后显示所有编辑操作按钮
- [x] 登出后界面恢复为游客状态（隐藏编辑功能）
