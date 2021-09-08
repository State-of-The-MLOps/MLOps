# INSERT
INSERT_MODEL_CORE = """
            INSERT INTO model_core (
                model_name,
                model_file
            ) VALUES(
                %s,
                '%s'
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
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
            """

# UPDATE
UPDATE_MODEL_METADATA = """
                    UPDATE model_metadata
                    SET 
                        train_mae = %s,
                        val_mae = %s,
                        train_mse = %s,
                        val_mse = %s
                    WHERE experiment_name = %s
                """
UPDATE_MODEL_CORE = """
                    UPDATE model_core
                    SET
                        model_file = '%s'
                    WHERE
                        model_name = %s
                """

# pd READ_SQL
SELECT_ALL_INSURANCE = """
            SELECT *
            FROM insurance
        """

SELECT_VAL_MAE = """
        SELECT val_mae
        FROM model_metadata
        WHERE model_core_name = %s
    """

SELECT_MODEL_CORE = """
        SELECT *
        FROM model_core
        WHERE model_name = %s
    """
