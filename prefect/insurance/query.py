INSERT_BEST_MODEL = """
        INSERT INTO best_model_data (
            model_name,
            artifact_uri,
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
            artifact_uri = '{}',
            model_type = '{}',
            metric = '{}',
            metric_score = {}
        WHERE
            model_name = '{}'
"""
