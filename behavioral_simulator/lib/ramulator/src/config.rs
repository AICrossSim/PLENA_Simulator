use anyhow::Result;

#[derive(serde::Deserialize)]
pub struct Config {
    #[serde(rename = "DRAM")]
    pub(crate) dram: serde_json::Value,
    #[serde(rename = "Controller")]
    pub(crate) controller: serde_json::Value,
    #[serde(rename = "AddrMapper")]
    pub(crate) addr_mapper: serde_json::Value,
}

impl Config {
    pub fn from_yaml(config: &str) -> Result<Self> {
        Ok(serde_yaml_ng::from_str(config)?)
    }
}
