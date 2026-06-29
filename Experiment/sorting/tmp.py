from cocube_udp import Soccer,CoCube
import time

soccer = Soccer(21, gateway='192.168.3.1', local_ip='192.168.3.118', ip_prefix=100)

# agent = CoCube(1, gateway='192.168.3.1', local_ip='192.168.3.118', ip_prefix=100)
# agent.gripper_open()
# time.sleep(1)
# agent.gripper_close()
# while True:
#     print(soccer.pos_p[0], soccer.pos_p[1])
#     # print(agent.pos_p[0], agent.pos_p[1])
#     time.sleep(1)

for i in range(1, 7):
    agent1 = CoCube(i, gateway='192.168.3.1', local_ip='192.168.3.118', ip_prefix=100)
    # agent1.move_to_target(150,100)
    # time.sleep(1)
    # agent2 = CoCube(2, gateway='192.168.3.1', local_ip='192.168.3.118', ip_prefix=100)
    agent1.wheels_break()
    # agent2.stop()