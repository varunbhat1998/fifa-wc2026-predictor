"""Country -> FIFA confederation lookup. Used as a categorical feature."""

UEFA = {
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
    "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
    "Czech Republic", "Czechoslovakia", "Denmark", "East Germany", "England",
    "Estonia", "Faroe Islands", "Finland", "France", "Georgia", "Germany",
    "Gibraltar", "Greece", "Hungary", "Iceland", "Ireland", "Israel", "Italy",
    "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein", "Lithuania",
    "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands",
    "North Macedonia", "Northern Ireland", "Norway", "Poland", "Portugal",
    "Republic of Ireland", "Romania", "Russia", "San Marino", "Scotland",
    "Serbia", "Serbia and Montenegro", "Slovakia", "Slovenia", "Soviet Union",
    "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine", "Wales",
    "West Germany", "Yugoslavia", "CIS",
}

CONMEBOL = {
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
    "Paraguay", "Peru", "Uruguay", "Venezuela",
}

CONCACAF = {
    "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados", "Belize", "Bermuda",
    "British Virgin Islands", "Canada", "Cayman Islands", "Costa Rica", "Cuba",
    "Curacao", "Dominica", "Dominican Republic", "El Salvador", "Grenada",
    "Guadeloupe", "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica",
    "Martinique", "Mexico", "Montserrat", "Netherlands Antilles", "Nicaragua",
    "Panama", "Puerto Rico", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Sint Maarten", "Suriname",
    "Trinidad and Tobago", "Turks and Caicos Islands", "United States",
    "US Virgin Islands", "French Guiana",
}

AFC = {
    "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan", "Brunei",
    "Cambodia", "China PR", "Chinese Taipei", "Guam", "Hong Kong", "India",
    "Indonesia", "Iran", "Iraq", "Japan", "Jordan", "Kuwait", "Kyrgyzstan",
    "Laos", "Lebanon", "Macau", "Malaysia", "Maldives", "Mongolia", "Myanmar",
    "Nepal", "North Korea", "Northern Mariana Islands", "Oman", "Pakistan",
    "Palestine", "Philippines", "Qatar", "Saudi Arabia", "Singapore",
    "South Korea", "Sri Lanka", "Syria", "Tajikistan", "Thailand", "Timor-Leste",
    "Turkmenistan", "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen",
}

CAF = {
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros",
    "Congo", "DR Congo", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea",
    "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
    "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya",
    "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco",
    "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe",
    "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa",
    "South Sudan", "Sudan", "Swaziland", "Tanzania", "Togo", "Tunisia",
    "Uganda", "Zaire", "Zambia", "Zanzibar", "Zimbabwe",
}

OFC = {
    "American Samoa", "Cook Islands", "Fiji", "Kiribati", "New Caledonia",
    "New Zealand", "Niue", "Papua New Guinea", "Samoa", "Solomon Islands",
    "Tahiti", "Tonga", "Tuvalu", "Vanuatu",
}

_ALL = {
    "UEFA": UEFA,
    "CONMEBOL": CONMEBOL,
    "CONCACAF": CONCACAF,
    "AFC": AFC,
    "CAF": CAF,
    "OFC": OFC,
}


def confederation_of(team: str) -> str:
    for name, members in _ALL.items():
        if team in members:
            return name
    return "OTHER"
