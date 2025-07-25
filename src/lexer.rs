// Copyright (C) 2025 Ryan Daum <ryan.daum@gmail.com> This program is free
// software: you can redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software Foundation, version
// 3.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// this program. If not, see <https://www.gnu.org/licenses/>.
//

//! Lexical analyzer for Lisp source code.
//! Converts text into a stream of tokens for parsing.

use std::fmt;

/// A token in the Lisp language
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// Left parenthesis: (
    LeftParen,
    /// Right parenthesis: )
    RightParen,
    /// Left bracket: [
    LeftBracket,
    /// Right bracket: ]
    RightBracket,
    /// Integer number: 42, -17
    Integer(i32),
    /// Floating point number: 3.14, -2.5
    Float(f64),
    /// Symbol: +, foo, my-var
    Symbol(String),
    /// Keyword: :foo, :my-keyword
    Keyword(String),
    /// String literal: "hello world"
    String(String),
    /// End of input
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::LeftParen => write!(f, "("),
            Token::RightParen => write!(f, ")"),
            Token::LeftBracket => write!(f, "["),
            Token::RightBracket => write!(f, "]"),
            Token::Integer(n) => write!(f, "{n}"),
            Token::Float(n) => write!(f, "{n}"),
            Token::Symbol(s) => write!(f, "{s}"),
            Token::Keyword(s) => write!(f, ":{s}"),
            Token::String(s) => write!(f, "\"{s}\""),
            Token::Eof => write!(f, "EOF"),
        }
    }
}

/// Lexical analyzer that converts source code into tokens
pub struct Lexer {
    input: Vec<char>,
    position: usize,
    current_char: Option<char>,
}

impl Lexer {
    /// Create a new lexer for the given input string
    pub fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let current_char = chars.first().copied();

        Self {
            input: chars,
            position: 0,
            current_char,
        }
    }

    /// Advance to the next character
    fn advance(&mut self) {
        self.position += 1;
        self.current_char = self.input.get(self.position).copied();
    }

    /// Skip whitespace characters
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Skip a comment (from ; to end of line)
    fn skip_comment(&mut self) {
        while let Some(ch) = self.current_char {
            if ch == '\n' {
                self.advance();
                break;
            }
            self.advance();
        }
    }

    /// Read a number (integer or float)
    fn read_number(&mut self) -> Result<Token, String> {
        let mut number_str = String::new();
        let mut is_float = false;

        // Handle negative sign
        if self.current_char == Some('-') {
            number_str.push('-');
            self.advance();
        }

        // Read digits and possibly a decimal point
        while let Some(ch) = self.current_char {
            if ch.is_ascii_digit() {
                number_str.push(ch);
                self.advance();
            } else if ch == '.' && !is_float {
                is_float = true;
                number_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        // Parse the number
        if is_float {
            number_str
                .parse::<f64>()
                .map(Token::Float)
                .map_err(|_| format!("Invalid float: {number_str}"))
        } else {
            number_str
                .parse::<i32>()
                .map(Token::Integer)
                .map_err(|_| format!("Invalid integer: {number_str}"))
        }
    }

    /// Read a symbol or keyword
    fn read_symbol(&mut self) -> Token {
        let mut symbol_str = String::new();

        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || "+-*/%=<>!?_-".contains(ch) {
                symbol_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        Token::Symbol(symbol_str)
    }

    /// Read a keyword (starting with :)
    fn read_keyword(&mut self) -> Token {
        // Skip the : character
        self.advance();

        let mut keyword_str = String::new();

        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || "-_".contains(ch) {
                keyword_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        Token::Keyword(keyword_str)
    }

    /// Read a string literal
    fn read_string(&mut self) -> Result<Token, String> {
        // Skip opening quote
        self.advance();

        let mut string_content = String::new();

        while let Some(ch) = self.current_char {
            if ch == '"' {
                // End of string
                self.advance();
                return Ok(Token::String(string_content));
            } else if ch == '\\' {
                // Handle escape sequences
                self.advance();
                match self.current_char {
                    Some('n') => string_content.push('\n'),
                    Some('t') => string_content.push('\t'),
                    Some('r') => string_content.push('\r'),
                    Some('\\') => string_content.push('\\'),
                    Some('"') => string_content.push('"'),
                    Some(other) => {
                        return Err(format!("Invalid escape sequence: \\{other}"));
                    }
                    None => {
                        return Err("Unexpected end of input in string escape".to_string());
                    }
                }
                self.advance();
            } else {
                string_content.push(ch);
                self.advance();
            }
        }

        Err("Unterminated string literal".to_string())
    }

    /// Get the next token from the input
    pub fn next_token(&mut self) -> Result<Token, String> {
        loop {
            match self.current_char {
                None => return Ok(Token::Eof),

                Some(ch) if ch.is_whitespace() => {
                    self.skip_whitespace();
                    continue;
                }

                Some(';') => {
                    self.skip_comment();
                    continue;
                }

                Some('(') => {
                    self.advance();
                    return Ok(Token::LeftParen);
                }

                Some(')') => {
                    self.advance();
                    return Ok(Token::RightParen);
                }

                Some('[') => {
                    self.advance();
                    return Ok(Token::LeftBracket);
                }

                Some(']') => {
                    self.advance();
                    return Ok(Token::RightBracket);
                }

                Some('"') => {
                    return self.read_string();
                }

                Some(':') => {
                    return Ok(self.read_keyword());
                }

                Some(ch) if ch.is_ascii_digit() => {
                    return self.read_number();
                }

                Some('-') => {
                    // Could be negative number or minus symbol
                    // Peek ahead to see if next char is a digit
                    if let Some(next_ch) = self.input.get(self.position + 1) {
                        if next_ch.is_ascii_digit() {
                            return self.read_number();
                        }
                    }
                    // Otherwise treat as symbol
                    return Ok(self.read_symbol());
                }

                Some(ch) if ch.is_alphabetic() || "+-*/%=<>!?_".contains(ch) => {
                    return Ok(self.read_symbol());
                }

                Some(ch) => {
                    return Err(format!("Unexpected character: '{ch}'"));
                }
            }
        }
    }

    /// Tokenize the entire input into a vector of tokens
    pub fn tokenize(&mut self) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();

        loop {
            let token = self.next_token()?;
            let is_eof = matches!(token, Token::Eof);
            tokens.push(token);

            if is_eof {
                break;
            }
        }

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let mut lexer = Lexer::new("( ) [ ] 42 3.14");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(
            tokens,
            vec![
                Token::LeftParen,
                Token::RightParen,
                Token::LeftBracket,
                Token::RightBracket,
                Token::Integer(42),
                Token::Float(3.14),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_symbols_and_keywords() {
        let mut lexer = Lexer::new("+ foo-bar :keyword :my-key");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(
            tokens,
            vec![
                Token::Symbol("+".to_string()),
                Token::Symbol("foo-bar".to_string()),
                Token::Keyword("keyword".to_string()),
                Token::Keyword("my-key".to_string()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_strings() {
        let mut lexer = Lexer::new(r#""hello world" "with\nescapes""#);
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(
            tokens,
            vec![
                Token::String("hello world".to_string()),
                Token::String("with\nescapes".to_string()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_negative_numbers() {
        let mut lexer = Lexer::new("-42 -3.14 - minus");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(
            tokens,
            vec![
                Token::Integer(-42),
                Token::Float(-3.14),
                Token::Symbol("-".to_string()),
                Token::Symbol("minus".to_string()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_comments() {
        let mut lexer = Lexer::new("42 ; this is a comment\n3.14");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(
            tokens,
            vec![Token::Integer(42), Token::Float(3.14), Token::Eof,]
        );
    }

    #[test]
    fn test_lisp_expression() {
        let mut lexer = Lexer::new("(+ (* 2 3) 4)");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(
            tokens,
            vec![
                Token::LeftParen,
                Token::Symbol("+".to_string()),
                Token::LeftParen,
                Token::Symbol("*".to_string()),
                Token::Integer(2),
                Token::Integer(3),
                Token::RightParen,
                Token::Integer(4),
                Token::RightParen,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_let_expression() {
        let mut lexer = Lexer::new("(let ((x 5)) (+ x :value))");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(
            tokens,
            vec![
                Token::LeftParen,
                Token::Symbol("let".to_string()),
                Token::LeftParen,
                Token::LeftParen,
                Token::Symbol("x".to_string()),
                Token::Integer(5),
                Token::RightParen,
                Token::RightParen,
                Token::LeftParen,
                Token::Symbol("+".to_string()),
                Token::Symbol("x".to_string()),
                Token::Keyword("value".to_string()),
                Token::RightParen,
                Token::RightParen,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_janet_style_let_expression() {
        let mut lexer = Lexer::new("(let [x 5 y 3] (+ x y))");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(
            tokens,
            vec![
                Token::LeftParen,
                Token::Symbol("let".to_string()),
                Token::LeftBracket,
                Token::Symbol("x".to_string()),
                Token::Integer(5),
                Token::Symbol("y".to_string()),
                Token::Integer(3),
                Token::RightBracket,
                Token::LeftParen,
                Token::Symbol("+".to_string()),
                Token::Symbol("x".to_string()),
                Token::Symbol("y".to_string()),
                Token::RightParen,
                Token::RightParen,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_error_unterminated_string() {
        let mut lexer = Lexer::new(r#""unterminated"#);
        let result = lexer.tokenize();

        assert!(result.is_err());
        assert!(result.err().unwrap().contains("Unterminated string"));
    }

    #[test]
    fn test_error_invalid_escape() {
        let mut lexer = Lexer::new(r#""\x""#);
        let result = lexer.tokenize();

        assert!(result.is_err());
        assert!(result.err().unwrap().contains("Invalid escape sequence"));
    }
}
