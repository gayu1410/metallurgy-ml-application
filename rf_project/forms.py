from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, DecimalField
from wtforms.validators import DataRequired


class predict_UTS_(FlaskForm):
    f_PcntCa = DecimalField('Percentage of Calcium',
                            validators=[DataRequired()])
    f_PcntC = DecimalField('Percentage of Carbon', validators=[DataRequired()])
    f_PcntNb = DecimalField('Percentage of Niobium',
                            validators=[DataRequired()])
    f_PcntMn = DecimalField('Percentage of Manganese',
                            validators=[DataRequired()])
    f_PcntN = DecimalField('Percentage of Nitrogen',
                           validators=[DataRequired()])

    submit = SubmitField('Predict UTS')


class predict_YS_(FlaskForm):
    f_PcntCa = DecimalField('Percentage of Calcium',
                            validators=[DataRequired()])
    f_PcntTi = DecimalField('Percentage of Titanium',
                            validators=[DataRequired()])
    f_PcntNb = DecimalField('Percentage of Niobium',
                            validators=[DataRequired()])
    f_PcntMn = DecimalField('Percentage of Manganese',
                            validators=[DataRequired()])
    slab_width = DecimalField('Slab Width', validators=[DataRequired()])

    submit = SubmitField('Predict YS')
