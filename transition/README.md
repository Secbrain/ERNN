# The Transition Matrix Settings for ERNN
According to some observations in our testbed, we provide proposal $F_m$ value ranges for a wired network in below Table. In Row1, when the current state is normal, the probabilities of normal, loss, retransmission, and out-of-order for the next state can be set within [0.6, 1], [0, 0.1], [0, 0.2], and [0, 0.1], respectively. When the current state is packet loss, packet retransmission and out-of-order may arise. Therefore, we set the probability range of four states as [0.5, 1], [0, 0.1], [0, 0.25], and [0, 0.15] in Row2. Similarly, retransmission may introduce some out-of-order packets, so in Row3 we set the range of normal, loss, retransmission, and out-of-order as [0.5, 1], [0, 0.1], [0, 0.2], and [0, 0.2], respectively. 
Finally, in Row4, we consider that the out-of-order packets usually affect their neighboring fragment. Thus we set the value range as [0.55, 1], [0, 0.1], [0, 0.2], and [0, 0.15], respectively.

<div align=center>
    <div style="color:orange; border-bottom: 1px solid #d9d9d7;
    display: inline-block;
    color: #999;
    padding: 2px;">Table. The proposal value ranges of session state update in the wired network.</div>

**States** | **$s_n$** | **$s_l$** | **$s_r$** | **$s_o$** 
:-: | :-: | :-: | :-: | :-: 
**$s_n$**   | $T_{nn} \in [ {0.6,1} ]$           | $T_{nl} \in [ {0,0.1} ]$           | $T_{nr} \in [ {0,0.2} ]$           | $T_{no} \in [ {0,0.1} ]$ 
**$s_l$**   | $T_{ln} \in [ {0.5,1} ]$           | $T_{ll} \in [ {0,0.1} ]$           | $T_{lr} \in [ {0,0.25} ]$           | $T_{lo} \in [ {0,0.15} ]$ 
**$s_r$**   | $T_{rn} \in [ {0.5,1} ]$           | $T_{rl} \in [ {0,0.1} ]$           | $T_{rr} \in [ {0,0.2} ]$           | $T_{rp} \in [ {0,0.2} ]$ 
**$s_o$**   | $T_{on} \in [ {0.55,1} ]$           | $T_{ol} \in [ {0,0.1} ]$           | $T_{or} \in [ {0,0.2} ]$           | $T_{oo} \in [ {0,0.15} ]$ 

</div>
