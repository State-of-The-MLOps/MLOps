# INSERT
INSERT_REG_MODEL = """
            INSERT INTO reg_model (
                model_name,
                path
            ) VALUES(
                %s,
                %s
            )
        """

INSERT_REG_MODEL_METADATA = """
                INSERT INTO reg_model_metadata (
                    experiment_name,
                    reg_model_name,
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
UPDATE_REG_MODEL_METADATA = """
                    UPDATE reg_model_metadata
                    SET 
                        train_mae = %s,
                        val_mae = %s,
                        train_mse = %s,
                        val_mse = %s
                    WHERE experiment_name = %s
                """

# pd READ_SQL
SELECT_ALL_INSURANCE = """
            SELECT *
            FROM insurance
        """

SELECT_VAL_MAE = """
        SELECT val_mae
        FROM reg_model_metadata
        WHERE reg_model_name = %s
    """

SELECT_REG_MODEL = """
        SELECT *
        FROM reg_model
        WHERE model_name = %s
    """
