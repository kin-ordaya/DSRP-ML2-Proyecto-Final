# 📖 Diccionario de Datos — Telco Customer Churn

## Fuente

- **Dataset**: IBM Telco Customer Churn
- **Origen**: [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Tamaño**: 7043 filas × 21 columnas
- **Formato**: CSV (`data/raw/telco_customer_churn.csv`)

## Descripción

Conjunto de datos tabular de una empresa de telecomunicaciones ficticia
que presta servicios de telefonía fija, internet y streaming. Cada fila
representa a un cliente con información demográfica, servicios contratados
y su historial de facturación. El objetivo es predecir la variable **Churn**
(abandono del servicio).

## Columnas

| Columna | Tipo | Descripción | Valores / Rango |
|---------|------|-------------|-----------------|
| `customerID` | `object` | Identificador único del cliente | 7043 valores únicos |
| `gender` | `object` | Género del cliente | `Male`, `Female` |
| `SeniorCitizen` | `int64` | Indica si el cliente es adulto mayor | `0` (No), `1` (Sí) |
| `Partner` | `object` | El cliente tiene pareja | `Yes`, `No` |
| `Dependents` | `object` | El cliente tiene dependientes | `Yes`, `No` |
| `tenure` | `int64` | Meses que el cliente lleva con el servicio | 0–72 |
| `PhoneService` | `object` | Suscripción al servicio telefónico | `Yes`, `No` |
| `MultipleLines` | `object` | Líneas telefónicas múltiples | `Yes`, `No`, `No phone service` |
| `InternetService` | `object` | Proveedor y tipo de servicio de internet | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | `object` | Servicio adicional de seguridad en línea | `Yes`, `No`, `No internet service` |
| `OnlineBackup` | `object` | Servicio adicional de respaldo en línea | `Yes`, `No`, `No internet service` |
| `DeviceProtection` | `object` | Servicio adicional de protección de dispositivo | `Yes`, `No`, `No internet service` |
| `TechSupport` | `object` | Servicio adicional de soporte técnico | `Yes`, `No`, `No internet service` |
| `StreamingTV` | `object` | Servicio adicional de TV en streaming | `Yes`, `No`, `No internet service` |
| `StreamingMovies` | `object` | Servicio adicional de películas en streaming | `Yes`, `No`, `No internet service` |
| `Contract` | `object` | Tipo de contrato del cliente | `Month-to-month`, `One year`, `Two year` |
| `PaperlessBilling` | `object` | Facturación sin papel (electrónica) | `Yes`, `No` |
| `PaymentMethod` | `object` | Método de pago | `Bank transfer (automatic)`, `Credit card (automatic)`, `Electronic check`, `Mailed check` |
| `MonthlyCharges` | `float64` | Monto mensual facturado al cliente | 18.25 – 118.75 (USD) |
| `TotalCharges` | `object` | Monto total acumulado (debe tratarse como numérico) | — (11 valores en blanco) |
| `Churn` | `object` | **Variable objetivo**: el cliente abandonó el servicio | `Yes`, `No` |

## Observaciones de calidad de datos

- **Valores nulos**: 0 valores `NaN`.
- **`TotalCharges`**: almacenada como `object`; debe convertirse a `float64`.
  Contiene 11 registros en blanco (espacios), asociados a clientes con
  `tenure == 0`.
- **Desbalance de clases**: la variable objetivo `Churn` tiene
  aproximadamente un 73% / 27% de distribución (No / Yes).
- Categorías especiales `No phone service` y `No internet service`: codifican
  la ausencia del servicio base y deben gestionarse en el preprocesamiento
  (se fusionan con `No` o se tratan como categoría propia).