from gym.envs.registration import register
 
register(id='TouchTestBlock-v0', 
         entry_point='gym_touchhand.envs:HandBlockTouchSensorsTestEnv',
         )

register(id='TouchTestEgg-v0', 
         entry_point='gym_touchhand.envs:HandEggTouchSensorsTestEnv',
         )

register(id='TouchTestPen-v0', 
         entry_point='gym_touchhand.envs:HandPenTouchSensorsTestEnv',
         )
