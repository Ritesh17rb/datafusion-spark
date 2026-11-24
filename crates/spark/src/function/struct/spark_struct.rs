// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{ArrayRef, StructArray};
use arrow::datatypes::{DataType, Field, FieldRef, Fields};
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, Expr, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, Volatility,
};
use datafusion_macros::user_doc;

/// Spark-compatible `struct` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#struct>
#[user_doc(
    doc_section(label = "Struct Functions"),
    description = "Creates a non-NULL struct, retaining NULL fields. Unnamed expressions use \
        one-based field names `col1`, `col2`, etc., rather than DataFusion's `c0`, `c1`, etc. \
        The Spark SQL planner and Rust expression helper retain column and nested field names. \
        The physical UDF and Rust helper accept zero arguments for an empty struct, \
        but DataFusion's SQL parser currently rejects `struct()`. \
        SQL calls mixing explicit aliases and unnamed arguments still receive DataFusion's \
        generated names for unnamed fields, because those names are assigned before the Spark planner.",
    syntax_example = "struct(expr, ...)",
    sql_example = r#"```sql
> SELECT struct(1, 2, 3);
{col1: 1, col2: 2, col3: 3}
```"#,
    argument(name = "expr", description = "Zero or more expressions of any type.")
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkStruct {
    signature: Signature,
    names: Vec<String>,
}

impl Default for SparkStruct {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkStruct {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
            names: vec![],
        }
    }

    /// Capture names before optimization folds expressions or changes columns.
    pub(crate) fn expression(args: Vec<Expr>) -> Expr {
        let names = args
            .iter()
            .enumerate()
            .map(|(i, expr)| {
                match expr {
                    Expr::Column(column) => Some(column.name.clone()),
                    Expr::Alias(alias) => Some(alias.name.clone()),
                    Expr::ScalarFunction(function)
                        if function.func.name() == "get_field" =>
                    {
                        match function.args.get(1) {
                            Some(Expr::Literal(ScalarValue::Utf8(Some(name)), _)) => {
                                Some(name.clone())
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                }
                .unwrap_or_else(|| format!("col{}", i + 1))
            })
            .collect();
        Arc::new(ScalarUDF::from(Self {
            names,
            ..Self::new()
        }))
        .call(args)
    }

    /// SQL has already interleaved names and values when any argument has an alias.
    pub(crate) fn named_expression(args: Vec<Expr>) -> Result<Expr> {
        let mut names = Vec::with_capacity(args.len() / 2);
        let mut values = Vec::with_capacity(args.len() / 2);
        let (pairs, remainder) = args.as_chunks::<2>();
        if !remainder.is_empty() {
            return internal_err!("struct planner expected name/value pairs");
        }
        for [name, value] in pairs {
            let Expr::Literal(ScalarValue::Utf8(Some(name)), _) = name else {
                return internal_err!("struct planner expected a literal field name");
            };
            names.push(name.clone());
            values.push(value.clone());
        }
        Ok(Arc::new(ScalarUDF::from(Self {
            names,
            ..Self::new()
        }))
        .call(values))
    }

    fn fields(&self, args: &[FieldRef]) -> Fields {
        args.iter()
            .enumerate()
            .map(|(i, field)| {
                let name = self
                    .names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("col{}", i + 1));
                Arc::new(field.as_ref().clone().with_name(name))
            })
            .collect()
    }
}

impl ScalarUDFImpl for SparkStruct {
    fn name(&self) -> &str {
        "struct"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        Ok(arg_types.to_vec())
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        let fields = arg_types
            .iter()
            .map(|t| Arc::new(Field::new("", t.clone(), true)))
            .collect::<Vec<_>>();
        Ok(DataType::Struct(self.fields(&fields)))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(Arc::new(Field::new(
            self.name(),
            DataType::Struct(self.fields(args.arg_fields)),
            false,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let DataType::Struct(fields) = args.return_type() else {
            return internal_err!("struct requires a struct return type");
        };
        if fields.len() != args.args.len() {
            return internal_err!(
                "struct return field count differs from argument count"
            );
        }
        let scalar = args
            .args
            .iter()
            .all(|arg| matches!(arg, ColumnarValue::Scalar(_)));
        let rows = if scalar { 1 } else { args.number_rows };
        let arrays = args
            .args
            .iter()
            .map(|arg| arg.to_array(rows))
            .collect::<Result<Vec<_>>>()?;
        let result: ArrayRef = Arc::new(if fields.is_empty() {
            StructArray::new_empty_fields(rows, None)
        } else {
            StructArray::try_new(fields.clone(), arrays, None)?
        });
        if scalar {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?))
        } else {
            Ok(ColumnarValue::Array(result))
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use datafusion_common::config::ConfigOptions;
    use datafusion_expr::{ExprSchemable, col, lit};

    #[test]
    fn field_names_nullability_and_metadata() -> Result<()> {
        let metadata = std::collections::HashMap::from([(
            "comment".to_string(),
            "input field".to_string(),
        )]);
        let input =
            Arc::new(Field::new("a", DataType::Int32, true).with_metadata(metadata));
        let udf = SparkStruct::new();
        let output = udf.return_field_from_args(ReturnFieldArgs {
            arg_fields: std::slice::from_ref(&input),
            scalar_arguments: &[None],
        })?;
        assert!(!output.is_nullable());
        let DataType::Struct(fields) = output.data_type() else {
            panic!("expected struct")
        };
        assert_eq!(
            fields[0].as_ref(),
            &input.as_ref().clone().with_name("col1")
        );

        let schema =
            datafusion_common::DFSchema::try_from(arrow::datatypes::Schema::new(vec![
                input,
            ]))?;
        let expr =
            SparkStruct::expression(vec![col("a"), lit(2).alias("renamed"), lit(3)]);
        let DataType::Struct(fields) = expr.get_type(&schema)? else {
            panic!("expected struct")
        };
        assert_eq!(
            fields.iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
            vec!["a", "renamed", "col3"]
        );
        assert!(fields[0].is_nullable());
        assert!(!fields[1].is_nullable());

        let named = SparkStruct::named_expression(vec![
            lit("c0"),
            lit(1),
            lit(""),
            lit(ScalarValue::Int32(None)),
        ])?;
        let DataType::Struct(fields) = named.get_type(&schema)? else {
            panic!("expected struct")
        };
        assert_eq!(fields[0].name(), "c0");
        assert!(!fields[0].is_nullable());
        assert_eq!(fields[1].name(), "");
        assert!(fields[1].is_nullable());
        Ok(())
    }

    #[test]
    fn physical_empty_batches_and_scalar_broadcast() -> Result<()> {
        let udf = SparkStruct::new();
        for values in [vec![], vec![Some(1), None]] {
            let rows = values.len();
            let inputs = vec![
                ColumnarValue::Array(Arc::new(Int32Array::from(values))),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(7))),
            ];
            let fields = vec![
                Arc::new(Field::new("a", DataType::Int32, true)),
                Arc::new(Field::new("b", DataType::Int32, false)),
            ];
            let output = udf.return_field_from_args(ReturnFieldArgs {
                arg_fields: &fields,
                scalar_arguments: &[None, None],
            })?;
            let result = udf.invoke_with_args(ScalarFunctionArgs {
                args: inputs,
                arg_fields: fields,
                number_rows: rows,
                return_field: output,
                config_options: Arc::new(ConfigOptions::default()),
            })?;
            let ColumnarValue::Array(array) = result else {
                panic!("expected array")
            };
            assert_eq!(array.len(), rows);
            assert_eq!(array.null_count(), 0);
            let structs = array.as_any().downcast_ref::<StructArray>().unwrap();
            assert_eq!(
                structs
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .as_ref(),
                vec![7; rows].as_slice()
            );
        }
        for rows in [0, 3] {
            let output = udf.return_field_from_args(ReturnFieldArgs {
                arg_fields: &[],
                scalar_arguments: &[],
            })?;
            assert!(!output.is_nullable());
            let result = udf.invoke_with_args(ScalarFunctionArgs {
                args: vec![],
                arg_fields: vec![],
                number_rows: rows,
                return_field: output,
                config_options: Arc::new(ConfigOptions::default()),
            })?;
            let array = result.to_array(rows)?;
            assert_eq!(array.len(), rows);
            assert_eq!(array.null_count(), 0);
            assert_eq!(array.data_type(), &DataType::Struct(Fields::empty()));
        }
        Ok(())
    }
}
