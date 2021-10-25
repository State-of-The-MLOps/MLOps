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

INSERT_OR_UPDATE_MODEL = """
UPDATE model_core
SET model_name='{mn}', model_file='{mf}'
WHERE model_core.model_name='{mn}';
INSERT INTO model_core (model_name, model_file)
SELECT '{mn}', '{mf}'
WHERE NOT EXISTS (SELECT 1 
				 FROM model_core as mc 
				 WHERE mc.model_name = '{mn}');
"""

INSERT_OR_UPDATE_SCORE = """
UPDATE atmos_model_metadata 
SET mae='{score1}', mse='{score2}'
WHERE atmos_model_metadata.model_name='{mn}';
INSERT INTO atmos_model_metadata (model_name, experiment_id, mae, mse)
SELECT '{mn}', '{expr_id}', '{score1}', '{score2}'
WHERE NOT EXISTS (SELECT 1 
				 FROM atmos_model_metadata as amm 
				 WHERE amm.model_name = '{mn}');
"""

SELECT_BEST_MODEL = """
    SELECT artifact_uri
    FROM best_model_data
    WHERE model_name = '{}'
"""
