INSERT_BEST_MODEL = """
        INSERT INTO best_model_data (
            model_name,
            run_id,
            model_type,
            metric,
            metric_score
        ) VALUES (
            '{}',
            '{}',
            '{}',
            '{}',
            {}
        )
    """

SELECT_EXIST_MODEL = """
        SELECT *
        FROM best_model_data
        WHERE model_name = '{}'
"""

UPDATE_BEST_MODEL = """
        UPDATE best_model_data
        SET
            run_id = '{}',
            model_type = '{}',
            metric = '{}',
            metric_score = {}
        WHERE
            model_name = '{}'
"""
