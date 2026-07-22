# Personal AI Workspace Platform

## 极简但可扩展 AI Agent 工作空间平台实现设计文档

版本：1.0

------------------------------------------------------------------------

# 1. 项目定位

本项目目标是构建一个 Web 化个人 AI 工作电脑。

每个用户拥有：

-   一个长期 Workspace
-   一个按需启动的 Agent Runtime
-   一个隔离 Sandbox
-   一个文件系统
-   一套 Skills
-   一套 Tools
-   一套 MCP 连接

用户体验类似：

-   Claude Projects
-   OpenHands Workspace
-   Cursor Workspace
-   Manus

核心理念：

> 每个用户拥有一个自己的 AI 工作电脑。

------------------------------------------------------------------------

# 2. 核心架构原则

## 2.1 Persistent Workspace

Workspace 永久存在。

保存：

-   文件
-   对话历史
-   Skills
-   Memory
-   用户配置

目录：

    workspace/

    ├── files/
    ├── outputs/
    ├── skills/
    ├── memory/
    └── temp/

------------------------------------------------------------------------

## 2.2 Ephemeral Runtime

Sandbox 是计算环境，不是数据存储。

生命周期：

    STOPPED

    ↓

    RUNNING

    ↓

    IDLE

    ↓

    STOPPED

用户离开：

-   停止 Docker
-   保留 Workspace

用户回来：

-   自动启动
-   恢复状态

------------------------------------------------------------------------

# 3. 总体架构

    Browser

        |

    Next.js Frontend

        |

    FastAPI Backend

        |

    ---------------------------------

    Auth

    Workspace

    Agent

    Sandbox Manager

    Skill Manager

    File Manager


        |

    PostgreSQL


        |

    Docker Engine


        |

    User Sandbox


        |

    smolagents Runtime


        |

    ----------------------------

    Skills

    Tools

    MCP

------------------------------------------------------------------------

# 4. 技术选型

## Frontend

-   Next.js
-   TypeScript
-   TailwindCSS
-   WebSocket

负责：

-   Chat
-   文件浏览
-   Skill管理
-   设置

------------------------------------------------------------------------

## Backend

FastAPI + Python

原因：

-   AI生态最佳
-   smolagents支持
-   MCP SDK支持

------------------------------------------------------------------------

## Database

PostgreSQL

保存：

-   用户
-   Workspace
-   消息
-   Skill
-   Sandbox状态

------------------------------------------------------------------------

## Storage

第一阶段：

本地文件系统。

不使用：

-   MinIO
-   S3

几百用户规模足够。

------------------------------------------------------------------------

## Agent Framework

smolagents

负责：

-   Agent Loop
-   Tool Calling
-   Reasoning

------------------------------------------------------------------------

## Sandbox

Docker。

负责：

-   环境隔离
-   代码执行
-   文件操作

------------------------------------------------------------------------

# 5. 用户模型

User:

    User

     |

    Workspace

     |

    Sandbox

     |

    Agent

------------------------------------------------------------------------

# 6. 数据库设计

## users

``` sql
users

id
email
password_hash
created_at
```

------------------------------------------------------------------------

## workspace

``` sql
workspace

id
user_id
name
path
sandbox_id
last_active
created_at
```

------------------------------------------------------------------------

## messages

``` sql
messages

id
workspace_id
role
content
created_at
```

role:

-   user
-   assistant
-   tool
-   system

------------------------------------------------------------------------

## api_credentials

``` sql
api_credentials

id
user_id
provider
encrypted_key
```

------------------------------------------------------------------------

## sandbox

``` sql
sandbox

id
workspace_id
container_id
status
last_active
```

------------------------------------------------------------------------

# 7. Workspace设计

Workspace是系统核心。

结构：

    /data/workspaces/user001/

    workspace/

    ├── files/
    ├── outputs/
    ├── skills/
    ├── memory/
    └── temp/

Docker挂载：

    Host:

    /data/workspaces/user001


            |

            |

    Container:

    /workspace

------------------------------------------------------------------------

# 8. Sandbox Manager

职责：

-   创建容器
-   启动容器
-   停止容器
-   查询状态

接口：

``` python
create_runtime()

start_runtime()

stop_runtime()

get_status()
```

------------------------------------------------------------------------

# 9. Sandbox自动休眠

规则：

例如：

60分钟无互动：

    RUNNING

    ↓

    IDLE

    ↓

    STOPPED

检测：

每5分钟执行：

``` python
if now-last_active > timeout:
    docker.stop()
```

------------------------------------------------------------------------

# 10. Agent Runtime

Sandbox内部：

    agent_runtime.py

            |

        smolagents

            |

    ----------------

    Tools

    Skills

    MCP

Agent负责：

-   理解用户意图
-   调用工具
-   修改文件
-   输出结果

------------------------------------------------------------------------

# 11. Chat系统

流程：

    User

    ↓

    Frontend

    ↓

    FastAPI

    ↓

    Workspace Agent

    ↓

    Sandbox

    ↓

    Agent Runtime

    ↓

    Response

实时通信：

WebSocket。

------------------------------------------------------------------------

# 12. 文件系统

文件属于Workspace。

不是Sandbox。

支持：

-   上传
-   删除
-   下载
-   Agent读取
-   Agent生成

------------------------------------------------------------------------

# 13. Skill系统

Skill:

    Skill

    =

    Prompt

    +

    Tools

    +

    Workflow

    +

    Knowledge

目录：

    skills/

    public/

    private/

示例：

    pdf-analysis/

    skill.yaml

    prompt.md

    tools/

------------------------------------------------------------------------

# 14. Tool系统

Tool是原子能力。

例如：

-   python_executor
-   file_reader
-   pdf_parser
-   image_processor
-   shell

接口：

``` python
class Tool:

    name

    description

    execute()
```

------------------------------------------------------------------------

# 15. MCP系统

MCP连接外部能力：

    Agent

     |

    MCP Client

     |

    MCP Server

例如：

-   Github
-   Notion
-   Google Drive
-   Database

------------------------------------------------------------------------

# 16. Memory系统

三层：

## Conversation Memory

数据库messages。

## Workspace Memory

    memory/state.json

保存任务状态。

## User Memory

    memory/profile.json

保存用户偏好。

------------------------------------------------------------------------

# 17. API Key BYOK

用户提供自己的：

-   OpenAI Key
-   Anthropic Key
-   DeepSeek Key

服务器：

只保存加密后的Key。

Agent执行：

    Agent

    ↓

    User API Key

    ↓

    LLM Provider

------------------------------------------------------------------------

# 18. 项目目录

    ai-workspace/

    frontend/

    backend/

        api/

        services/

        models/


    sandbox/

        runtime/

        tools/


    skills/

    data/


    docker-compose.yml

------------------------------------------------------------------------

# 19. 单服务器部署

    Server

    |

    Docker Compose


    ├── frontend

    ├── backend

    ├── postgres

    └── user containers

------------------------------------------------------------------------

# 20. 资源估算

目标：

几百用户。

推荐：

CPU: 16-32 Core

RAM: 64GB

SSD: 1TB

------------------------------------------------------------------------

# 21. 第一阶段不做

不要：

-   Kubernetes
-   微服务
-   Kafka
-   Redis
-   Celery
-   MinIO
-   Vector Database
-   Multi Agent

------------------------------------------------------------------------

# 22. 开发路线

## Phase 1

完成：

-   用户
-   Workspace
-   Chat
-   文件
-   Agent

## Phase 2

加入：

-   Docker Sandbox
-   自动停止
-   自动恢复

## Phase 3

加入：

-   Skills Marketplace
-   MCP
-   Tool生态

------------------------------------------------------------------------

# 23. 最终架构

    User

     |

    Workspace

     |

    Sandbox

     |

    Agent

     |

    Skills + Tools + MCP

------------------------------------------------------------------------

# 24. 产品定义

> 一个让每个人拥有自己的 AI 工作电脑的平台。

Workspace永久保存。

Sandbox按需启动。

Agent通过Skills、Tools和MCP无限扩展能力。
