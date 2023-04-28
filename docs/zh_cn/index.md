---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "EdgeLab"
  text: "Seeed Studio EdgeLab 是一个专注于嵌入式人工智能的开源项目。"
  tagline: 我们针对现实世界场景对许多算法进行了优化，使其实现更加人性化，并在嵌入式设备上实现了更快、更准确的推理。
  image:
    src: /images/EdgeLab-Hero.png
    alt: EdgeLab
  actions:
    - theme: brand
      text: 入门指南
      link: ./introduction/quick_start
    - theme: alt
      text: 安装 EdgeLab
      link: ./introduction/installation
    - theme: alt
      text: 查看 GitHub 仓库
      link: https://github.com/Seeed-Studio/EdgeLab

features:
  - icon: 🔍
    title: 异常检测 (即将推出)
    details: 在现实世界中，异常数据往往很难被识别，即使能被识别，也需要很高的成本。异常检测算法以低成本的方式收集正常数据，任何超出正常数据的东西都被认为是异常的。  
  - icon: 👁️
    title: 计算机视觉
    details: EdgeLab 提供了一些计算机视觉算法，如物体检测、图像分类、图像分割和姿态估计。然而，这些算法不能在低成本的硬件上运行。我们对这些计算机视觉算法进行了优化，以便在低端设备中实现良好的运行速度和准确性。 
  - icon: ⏱️
    title: 场景定制
    details: EdgeLab 为特定的生产环境提供定制场景的解决方案，例如模拟仪表、传统数字仪表的读数和音频分类。我们将在未来继续为特定场景添加更多的算法支持。敬请关注!
---

