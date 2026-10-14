# hello-gpu 笔记

Datawhale 85 期 hello-gpu 组队学习的要点式笔记，用 [VitePress](https://vitepress.dev/) 生成，推到 `main` 后由 GitHub Actions 自动发布到 GitHub Pages。

- 站点源码在 `docs/`，配置在 `docs/.vitepress/config.mts`
- 笔记正文在 `docs/notes/`
- 首页是 `docs/index.md`

## 本地预览（需要 Node 18+）

```bash
npm install
npm run docs:dev      # 本地热更新预览
npm run docs:build    # 构建到 docs/.vitepress/dist
```

## 第一次发布到 GitHub Pages

`base` 目前写成 `/hello-gpu-notes/`，对应仓库名 `hello-gpu-notes`。**如果仓库名不是这个，先改 `docs/.vitepress/config.mts` 里的 `base`，再往下走。**

1. 在 GitHub 上新建一个**空仓库**（不要勾选 Add README / .gitignore），仓库名取 `hello-gpu-notes`（或你改过的名字）。
2. 本目录已经 `git init` 并做好首个 commit（分支 `main`），所以只需接上远端再推：

   ```bash
   git remote add origin git@github.com:<你的用户名>/hello-gpu-notes.git
   git push -u origin main
   ```

   （若你拿到的是未初始化的副本，则改为：`git init && git add . && git commit -m "chore: 初始化 hello-gpu 笔记站" && git branch -M main`，再接上面的 remote + push。）

3. 打开仓库 **Settings → Pages**，把 **Source** 设为 **GitHub Actions**。
4. 等 Actions 跑完，站点地址是 `https://<你的用户名>.github.io/hello-gpu-notes/`。

## 新增一篇笔记

1. 在 `docs/notes/` 新建 `.md`。
2. 在 `docs/.vitepress/config.mts` 的 `sidebar` 里加一行 `{ text: '...', link: '/notes/文件名' }`。
3. commit + push。
