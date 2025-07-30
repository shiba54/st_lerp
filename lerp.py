import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
import streamlit as st


def callback_apply_edited_rows(
        key_data_editor: str,
        key_target: str
    ) -> None:
    """
    Apply edited rows to target dataframe

    Parameters
    --------
    key_data_editor : str
        specified key at st.data_editor
    key_target : str
        key in st.session_state
        st.session_state[key_target] must be dataframe
    """
    dict_edited_rows = st.session_state[key_data_editor]['edited_rows']
    for idx, dict_edited_row in dict_edited_rows.items():
            for col, val in dict_edited_row.items():
                st.session_state[key_target].loc[idx, col] = val


def callback_change_rows(
        key_number_input: str,
        key_df: str
    ) -> None:
    """
    Change number of rows

    Parameters
    --------
    key_num : str
        specified key at st.number_input
    key_df : str
        key in st.session_state
        st.session_state[key_target] must be dataframe
    """
    num: int = st.session_state[key_number_input]
    df: pd.DataFrame = st.session_state[key_df]

    if num < len(df):
        # Remove rows
        st.session_state[key_df] = df[: num]
    elif num > len(df):
        # Add rows
        num_add = num - len(df)
        data_add = {col: [0 for _ in range(num_add)] for col in df.columns}
        df_add = pd.DataFrame(data_add)
        st.session_state[key_df] = pd.concat([df, df_add])

    st.session_state[key_df].reset_index(drop=True, inplace=True)


def init_df_xy(
        key_df: str,
        num: int
    ) -> None:
    """
    Initialize df in st.session_state

    Parameters
    --------
    key_df : str
        key in st.session_state
        st.session_state[key_target] must be dataframe
    num : int
        number of rows
    """
    data_xy = {
        'x': [0.0 for _ in range(num)],
        'y': [0.0 for _ in range(num)]
    }
    df_xy = pd.DataFrame(data_xy)
    st.session_state[key_df] = df_xy


def main():
    st.set_page_config(
        page_title='Lerp',
        page_icon='☕',
        layout='wide'
    )
    st.title('Lerp')
    st.write('線形補間アプリ')

    col_xy, col_x_new = st.columns(2, border=True)

    # Sample point
    with col_xy:
        st.write(':memo: 既知点の設定')
        st.write(':material/Check: 点の数')
        st.number_input(
            label='_',
            min_value=2,
            value=len(st.session_state['df_xy']) if 'df_xy' in st.session_state else 2,
            step=1,
            key='num_xy',
            on_change=callback_change_rows,
            args=('num_xy', 'df_xy'),
            label_visibility='collapsed'
        )

        number_column = st.column_config.NumberColumn(
            label=None,
            required=True,
            default=0.0
        )
        column_config_xy = {
            'x': number_column,
            'y': number_column
        }

        if 'df_xy' not in st.session_state:
            init_df_xy(key_df='df_xy', num=2)

        st.write(':material/Check: 点の xy')
        st.data_editor(
            data=st.session_state['df_xy'],
            hide_index=True,
            column_config=column_config_xy,
            num_rows='fixed',
            key='edited_xy',
            on_change=callback_apply_edited_rows,
            args=('edited_xy', 'df_xy')
        )

        x = st.session_state['df_xy'].loc[:, 'x']
        y = st.session_state['df_xy'].loc[:, 'y']

        if x.is_monotonic_decreasing:
            # Make x increasing to use make_interp_spline
            st.session_state['df_xy'] = st.session_state['df_xy'] \
                .sort_values('x', ascending=True) \
                .reset_index(drop=True)
            x = st.session_state['df_xy'].loc[:, 'x']

        if x.nunique() < len(x):
            st.error('x に重複があります')
            can_interp = False
        elif not x.is_monotonic_increasing:
            st.error('x は昇順または降順としてください')
            can_interp = False
        else:
            can_interp = True

    # Interpolation point
    with col_x_new:
        st.write(':memo: 補間する点の設定')
        st.write(':material/Check: 指定方法')

        is_equal_interval = st.radio(
            label='_',
            options=[True, False],
            format_func=lambda x: '間隔を指定' if x else '任意の点を指定',
            on_change=init_df_xy,
            args=('df_xy_new', 2),
            label_visibility='collapsed'
        )

        if is_equal_interval:
            st.write(':material/Check: 始点の x')
            x_begin = st.number_input(
                label='_',
                label_visibility='collapsed',
                key='x_begin'
            )
            st.write(':material/Check: 終点の x')
            x_end = st.number_input(
                label='_',
                label_visibility='collapsed',
                key='x_end'
            )
            st.write(':material/Check: 間隔')
            x_pitch = st.number_input(
                label='_',
                label_visibility='collapsed',
                key='x_pitch'
            )

            try:
                x_new = np.arange(x_begin, x_end + x_pitch, x_pitch)
            except ZeroDivisionError:
                x_new = [0.0]

            data_xy_new = {
                'x': x_new,
                'y': [0.0 for _ in range(len(x_new))]
            }
            df_xy_new = pd.DataFrame(data_xy_new)
            st.session_state['df_xy_new'] = df_xy_new

        else:
            st.write(':material/Check: 点の数')
            st.number_input(
                label='_',
                min_value=1,
                value=2,
                step=1,
                key='num_new',
                on_change=callback_change_rows,
                args=('num_new', 'df_xy_new'),
                label_visibility='collapsed'
            )

            st.write(':material/Check: 点の x')
            st.data_editor(
                data=st.session_state['df_xy_new'],
                hide_index=True,
                column_config=column_config_xy,
                column_order=['x'],
                num_rows='fixed',
                key='edited_xy_new',
                on_change=callback_apply_edited_rows,
                args=('edited_xy_new', 'df_xy_new')
            )

    if can_interp:
        # Result
        spl = make_interp_spline(x, y, k=1)  # Linear
        y_new = spl(st.session_state['df_xy_new'].loc[:, 'x'])
        st.session_state['df_xy_new'].loc[:, 'y'] = y_new

        with st.container(border=True):

            col_dataframe, col_chart = st.columns([0.3, 0.7])

            with col_dataframe:
                st.write(':sparkles: 補間された点')
                st.dataframe(
                    data=st.session_state['df_xy_new'],
                    hide_index=True
                )
                st.download_button(
                    label='Download CSV',
                    data=st.session_state['df_xy_new'].to_csv(index=False),
                    file_name='lerp.csv',
                    on_click='ignore'
                )
            with col_chart:
                layout = go.Layout(
                    xaxis=dict(title='x'),
                    yaxis=dict(title='y'),
                    showlegend=True,
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    )
                )
                fig = go.Figure(layout=layout)
                fig.add_trace(go.Scatter(
                    x=st.session_state['df_xy'].loc[:, 'x'],
                    y=st.session_state['df_xy'].loc[:, 'y'],
                    mode='markers',
                    marker=dict(
                        color='grey',
                        size=8,
                    ),
                    hovertemplate='x: %{x}<br>y: %{y}',
                    name='既知点'
                ))
                fig.add_trace(go.Scatter(
                    x=st.session_state['df_xy'].loc[:, 'x'],
                    y=st.session_state['df_xy'].loc[:, 'y'],
                    mode='lines',
                    line=dict(
                        color='grey',
                        width=1
                    ),
                    hovertemplate='x: %{x}<br>y: %{y}',
                    name='既知点間'
                ))
                fig.add_trace(go.Scatter(
                    x=st.session_state['df_xy_new'].loc[:, 'x'],
                    y=st.session_state['df_xy_new'].loc[:, 'y'],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=5
                    ),
                    hovertemplate='x: %{x}<br>y: %{y}',
                    name='補間された点'
                ))
                st.plotly_chart(fig)

    st.markdown("""
    * ブラウザ更新でリセットできます
    * 表の値はコピペできます
    * 1次元のみ対象としています
    * 既知点の区間外も補間されます
    """)


if __name__ == '__main__':
    main()
