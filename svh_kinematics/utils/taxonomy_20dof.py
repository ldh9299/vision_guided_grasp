
grasp_dict_20f = {
    "Parallel_Extension": {
        'joint_init':  [10, 5, 5, 5,
                        0, 5, 5, 5,
                        0, 5, 5, 5,
                        0, 5, 5, 5,
                        0, 5, 5, 5],

        # 'joint_angle': [10, 6, 6, 6,
        #                 0, 6, 6, 6,
        #                 0, 6, 6, 6,
        #                 0, 6, 6, 6,
        #                 0, 6, 6, 6],

        'joint_final': [10, 40, 5, 5,
                        0, 25, 5, 5,
                        0, 25, 5, 5,
                        0, 25, 5, 5,
                        0, 25, 5, 5],
    },

    "Pen_Pinch": {
        'joint_init': [8,   15,  1,  1,
                       10,  1,  35,  35,
                       0,   1,  40,  40,
                       0,  60, 65, 65,
                       -5, 60, 65, 65],

        # 'joint_angle': [6, 15, 5, 5,
        #                 3, 5, 35, 35,
        #                 0, 7, 40, 40,
        #                 0, 60, 65, 65,
        #                 -3, 60, 65, 65],

        'joint_final': [8,  30,   5,   5,
                        5,  10,   45,  45,
                        0,  10,  50,  50,
                        0,  60,  65,  65,
                        -5, 60,  65,  65],
    },

    "Palmar_Pinch": {
        'joint_init': [5,  5,  5,  5,
                       5,  10,  5,  5,
                       5,  10,  5,  5,
                       0, 60, 65, 65,
                       -5, 60, 65, 65],

        # "joint_angle": [5, 15, 5, 5,
        #                 5, 15, 5, 5,
        #                 5, 15, 5, 5,
        #                 0, 60, 65, 65,
        #                 -5, 60, 65, 65],

        "joint_final": [5, 25, 5, 5,
                        5, 25, 15, 15,
                        5, 25, 20, 20,
                        0, 60, 65, 65,
                        -5, 60, 65, 65]
    },

    "Precision_Sphere": {
        'joint_init': [15,  2,  2,  2,
                       15,  2,  2,  2,
                       5,   2,  2,  2,
                       -5,  2,  2,  2,
                       -15, 2,  2,  2],

        # "joint_angle": [15, 5, 10, 10,
        #                 15, 5, 10, 10,
        #                 10,  5, 10, 10,
        #                 -5,  5, 10, 10,
        #                 -10, 5, 10, 10],

        "joint_final": [15,  10, 30, 30,
                        15,  30, 65, 65,
                        5,   45, 65, 65,
                        -5,  45, 55, 55,
                        -15, 45, 55, 55]
    },

    'Large_Wrap': {
        'joint_init': [15,  2,  2,  2,
                       -5,  2,  2,  2,
                       -8,  2,  2,  2,
                       -10, 2,  2,  2,
                       -10, 2,  2,  2],

        # 'joint_angle': [10, 3, 5, 5,
        #                 -5, 29, 51, 49,
        #                 -5, 34, 50, 50,
        #                 -7, 36, 48, 47,
        #                 -8, 31, 49, 51],

        'joint_final': [10, 12, 30, 30,
                        -5, 30, 65, 65,
                        -8, 35, 65, 65,
                        -10, 30, 65, 65,
                        -10, 25, 60, 60]
    },
}
