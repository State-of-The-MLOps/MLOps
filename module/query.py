# INSERT
INSERT_REG_MODEL = """
            INSERT INTO reg_model (
                model_name,
                path
            ) VALUES(
                {},
                {}
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
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        {}
                    )
            """

# UPDATE
UPDATE_REG_MODEL_METADATA = """
                    UPDATE reg_model_metadata
                    SET 
                        train_mae = {},
                        val_mae = {},
                        train_mse = {},
                        val_mse = {}
                    WHERE experiment_name = {}
                """

# pd READ_SQL
ALL_INSURANCE = """
            SELECT *
            FROM insurance
        """

VAL_MAE = """
        SELECT val_mae
        FROM reg_model_metadata
        WHERE reg_model_name = %s
    """

MODEL = """
        SELECT *
        FROM reg_model
        WHERE model_name = %s
    """
