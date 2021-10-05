UPDATE_TEMP_MODEL_DATA = """
            DELETE FROM temp_model_data
            WHERE id NOT IN (
                SELECT id
                FROM temp_model_data
                WHERE experiment_name = '{}'
                ORDER BY {}
                LIMIT {}
            )
        """


SELECT_TEMP_MODEL_BY_EXPR_NAME = """
            SELECT * 
            FROM temp_model_data
            WHERE experiment_name = '{}'
            ORDER BY {};
        """


SELECT_MODEL_METADATA_BY_EXPR_NAME = """
            SELECT *
            FROM model_metadata
            WHERE experiment_name = '{}'
        """

INSERT_MODEL_CORE = """
                INSERT INTO model_core (
                    model_name,
                    model_file
                ) VALUES(
                    '{}',
                    '{}'
                )
            """

INSERT_MODEL_METADATA = """
                INSERT INTO model_metadata (
                    experiment_name,
                    model_core_name,
                    experimenter,
                    version,
                    train_mae,
                    val_mae,
                    train_mse,
                    val_mse
                ) VALUES (
                    '{}',
                    '{}',
                    '{}',
                    '{}',
                    '{}',
                    '{}',
                    '{}',
                    '{}'
                )
"""

UPDATE_MODEL_CORE = """
            UPDATE model_core
            SET
                model_file = '{}'
            WHERE
                model_name = '{}'
        """

UPDATE_MODEL_METADATA = """
            UPDATE model_metadata
            SET 
                train_mae = {},
                val_mae = {},
                train_mse = {},
                val_mse = {}
            WHERE experiment_name = '{}'
        """

DELETE_ALL_EXPERIMENTS_BY_EXPR_NAME = """
    DELETE FROM temp_model_data
    WHERE experiment_name = '{}'
"""
