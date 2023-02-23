psic变化时的reachable set不再是4个polygon(平面)组成，而是4个polytope(体积)组成；
同样是每个polytope代表从某个接触面推动时的reachable states；
每个polytope的形成方式：假设当前psic=psic0，变化范围psic∈[psic0-dpsic_max, psic0+dpsic_max]，这个范围内的每一个psic对应的reachable states是一个polygon。容易证明，不同的psic对应的polygon，其相应顶点只在theta轴上会有移动，而xy坐标相同，因此这些polygon沿theta轴扫出一个体积，就是我们的polytope；
黄色点是随机采样的状态，蓝色点是从reachable set找到的最近邻，粉红色点是实际到达的状态点；

这个可视化的reachable set对应的state，我们假设接触面已知，则在该接触面，psic0可以不为0，在其他3个接触面，psic0=0(即默认从面的中心开始推动)；
