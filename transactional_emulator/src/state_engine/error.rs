use std::fmt;

use super::descriptor::DescriptorError;
use super::generated_contract::StateStatus;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateEngineError {
    pub status: StateStatus,
    message: String,
}

impl StateEngineError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(StateStatus::InvalidDescriptor, message)
    }

    pub fn address(message: impl Into<String>) -> Self {
        Self::new(StateStatus::AddressError, message)
    }

    pub fn hazard(message: impl Into<String>) -> Self {
        Self::new(StateStatus::StateHazard, message)
    }

    pub fn unsupported_precision(message: impl Into<String>) -> Self {
        Self::new(StateStatus::UnsupportedPrecision, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StateStatus::InternalError, message)
    }

    fn new(status: StateStatus, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl From<DescriptorError> for StateEngineError {
    fn from(error: DescriptorError) -> Self {
        Self::new(error.status, error.to_string())
    }
}

impl fmt::Display for StateEngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for StateEngineError {}
