// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Spark struct functions.

use datafusion_expr::ScalarUDF;
use datafusion_functions::make_udf_function;
use std::sync::Arc;

pub mod spark_struct;

make_udf_function!(spark_struct::SparkStruct, r#struct);

pub mod expr_fn {
    use datafusion_expr::Expr;

    /// Creates a Spark struct, retaining column names and assigning `colN` to expressions.
    pub fn r#struct(args: Vec<Expr>) -> Expr {
        super::spark_struct::SparkStruct::expression(args)
    }
}

pub fn functions() -> Vec<Arc<ScalarUDF>> {
    vec![r#struct()]
}
