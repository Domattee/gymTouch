from gym.envs.registration import register
 
register(id='TouchTestBlock-v0', 
         entry_point='gymTouchTestEnv.envs:HandBlockTouchSensorsTestEnv',
         )

register(id='TouchTestEgg-v0', 
         entry_point='gymTouchTestEnv.envs:HandEggTouchSensorsTestEnv',
         )

register(id='TouchTestPen-v0', 
         entry_point='gymTouchTestEnv.envs:HandPenTouchSensorsTestEnv',
         )
